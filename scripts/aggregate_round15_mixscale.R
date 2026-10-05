# Two passes over DEVELOPMENT columns only: NT means, then matched guide means.
# Input RDS is deserialized in RAM, but reserved expression columns are never selected.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=5) stop('Usage: input.rds mapping_dir output_dir min_controls chunk_cells')
for (pkg in c('SeuratObject','Matrix','jsonlite')) {
  if (!requireNamespace(pkg, quietly=TRUE)) stop(paste('Missing package',pkg))
}
mapping <- args[2]; out <- args[3]
min_controls <- as.integer(args[4]); chunk <- as.integer(args[5])
if (is.na(min_controls) || min_controls<1 || is.na(chunk) || chunk<1) stop('Invalid counts')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
progress <- function(stage,done,total) {
  cat(sprintf('MIXSCALE ADAPT %s cells=%d/%d\n',stage,done,total)); flush.console()
  jsonlite::write_json(list(stage=stage,done=done,total=total),file.path(out,'progress.json'),auto_unbox=TRUE)
}
cells <- read.csv(gzfile(file.path(mapping,'cells.csv.gz')),colClasses='character',check.names=FALSE)
genes <- readLines(file.path(mapping,'panel.txt'))
source_genes <- readLines(file.path(mapping,'source_panel.txt'))
if (length(source_genes)!=length(genes) || anyDuplicated(source_genes) || any(!nzchar(source_genes))) stop('Invalid one-to-one output mapping')
groups <- read.csv(file.path(mapping,'groups.csv'),check.names=FALSE)
strata <- read.csv(file.path(mapping,'strata.csv'),colClasses='character',check.names=FALSE)
if (any(cells$cell_type=='HT29') || anyDuplicated(cells$cell_id)) stop('Reserved/duplicate cells in manifest')
obj <- readRDS(args[1]); md <- obj[[]]
if (!inherits(obj,'Seurat')) stop('Expected Seurat object')
if (!setequal(cells$cell_id,rownames(md)[md$cell_type!='HT29'])) stop('Manifest must contain exactly all non-HT29 cells')
idx <- match(cells$cell_id,rownames(md))
for (field in c('cell_type','sample','pathway','Batch_info','orig.ident','sample_ID','gene','guide')) {
  if (anyNA(md[[field]][idx]) || !identical(as.character(md[[field]][idx]),cells[[field]])) stop(paste('Metadata changed:',field))
}
assay <- obj[['RNA']]
if (inherits(assay,'Assay5')) {
  layers <- SeuratObject::Layers(assay)
  if (sum(grepl('^counts($|\\.)',layers))!=1 || !'counts' %in% layers) stop('Exactly one RNA counts layer required')
  counts <- SeuratObject::LayerData(assay,layer='counts')
} else counts <- methods::slot(assay,'counts')
if (!inherits(counts,'sparseMatrix')) stop('Refusing dense counts')
if (anyDuplicated(rownames(counts)) || anyDuplicated(colnames(counts))) stop('Duplicate matrix identifiers')
cols <- match(cells$cell_id,colnames(counts)); gi <- match(source_genes,rownames(counts))
if (anyNA(cols) || anyNA(gi)) stop('Missing cells or panel genes')
s <- as.integer(cells$stratum_id); g <- as.integer(cells$group_id)
control <- cells$is_control=='1'; ns <- nrow(strata); ng <- nrow(groups); p <- length(genes)
if (!identical(sort(unique(s)),seq_len(ns)) || !identical(sort(unique(g)),seq_len(ng))) stop('Noncontiguous mapping IDs')
# Double precision sums, bounded panel-by-stratum and panel-by-guide arrays.
if (8*p*(ns+2*ng)+4*ns*ng>4*1024^3) stop('Aggregate arrays exceed 4 GiB; review strata before proceeding')
ctrl_sum <- matrix(0,p,ns); ctrl_n <- integer(ns)
mean_sum <- matrix(0,p,ng); base_sum <- matrix(0,p,ng)
n_input <- tabulate(g,nbins=ng); n_positive <- integer(ng); n_matched <- integer(ng)
used_strata <- matrix(FALSE,ns,ng)
read_chunk <- function(rows) {
  x <- counts[,cols[rows],drop=FALSE]
  if (any(!is.finite(x@x)) || any(x@x<0) || any(abs(x@x-round(x@x))>1e-5)) stop('Invalid raw counts')
  total <- Matrix::colSums(x)
  good <- total>0; rows <- rows[good]; x <- x[gi,good,drop=FALSE]
  x <- x %*% Matrix::Diagonal(x=10000/total[good])
  x@x <- log1p(x@x)
  list(rows=rows,x=x)
}
controls <- which(control)
if (!length(controls)) stop('No development NT controls')
for (start in seq.int(1,length(controls),by=chunk)) {
  stop_at <- min(start+chunk-1,length(controls)); z <- read_chunk(controls[start:stop_at])
  for (id in unique(s[z$rows])) {
    take <- which(s[z$rows]==id)
    ctrl_sum[,id] <- ctrl_sum[,id]+Matrix::rowSums(z$x[,take,drop=FALSE])
    ctrl_n[id] <- ctrl_n[id]+length(take)
  }
  progress('controls',stop_at,length(controls))
}
ctrl_mean <- sweep(ctrl_sum,2,pmax(ctrl_n,1),'/'); rm(ctrl_sum)
strata$n_control_input <- tabulate(s[control],nbins=ns)
strata$n_control_positive <- ctrl_n
strata$eligible <- ctrl_n>=min_controls
write.csv(strata,file.path(out,'control_qc.csv'),row.names=FALSE)
for (start in seq.int(1,nrow(cells),by=chunk)) {
  stop_at <- min(start+chunk-1,nrow(cells)); z <- read_chunk(seq.int(start,stop_at))
  n_positive <- n_positive+tabulate(g[z$rows],nbins=ng)
  eligible <- ctrl_n[s[z$rows]]>=min_controls
  z$x <- z$x[,eligible,drop=FALSE]; z$rows <- z$rows[eligible]
  for (id in unique(g[z$rows])) {
    take <- which(g[z$rows]==id); ss <- s[z$rows[take]]
    mean_sum[,id] <- mean_sum[,id]+Matrix::rowSums(z$x[,take,drop=FALSE])
    # Preserve the perturbation cells' stratum proportions in their control baseline.
    weights <- tabulate(ss,nbins=ns); used <- which(weights>0)
    used_strata[used,id] <- TRUE
    base_sum[,id] <- base_sum[,id]+as.vector(ctrl_mean[,used,drop=FALSE] %*% weights[used])
    n_matched[id] <- n_matched[id]+length(take)
  }
  progress('targets',stop_at,nrow(cells))
}
for (name in c('mean','baseline')) {
  sums <- if (name=='mean') mean_sum else base_sum
  mat <- t(sweep(sums,2,pmax(n_matched,1),'/'))
  df <- data.frame(group_id=seq_len(ng),mat,check.names=FALSE); names(df) <- c('group_id',genes)
  con <- gzfile(file.path(out,paste0(name,'.csv.gz')),'wt'); write.csv(df,con,row.names=FALSE); close(con)
}
write.csv(data.frame(group_id=seq_len(ng),n_input=n_input,n_positive=n_positive,n_matched=n_matched),
          file.path(out,'group_qc.csv'),row.names=FALSE)
links <- which(used_strata,arr.ind=TRUE)
write.csv(data.frame(group_id=links[,2],stratum_id=links[,1],n_controls=ctrl_n[links[,1]]),
          file.path(out,'matched_strata.csv'),row.names=FALSE)
jsonlite::write_json(list(state='AGGREGATED',development_cells=nrow(cells),
  HT29_expression_exported=FALSE,normalization='mean(log1p(10000*counts/full_library))',
  target_sum=10000,match_fields=c('cell_type','pathway','Batch_info','orig.ident','sample_ID'),
  min_controls=min_controls,chunk_cells=chunk),file.path(out,'aggregation.json'),pretty=TRUE,auto_unbox=TRUE)
progress('complete',nrow(cells),nrow(cells))
