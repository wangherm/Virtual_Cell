# Library QC only: all development NT cells plus four HAP1 perturbations.
# Reserved expression columns and other perturbation expression are never selected.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=3) stop('Usage: input.rds selected_cells.csv output_dir')
for (pkg in c('SeuratObject','Matrix','jsonlite')) if (!requireNamespace(pkg,quietly=TRUE)) stop(paste('Missing',pkg))
out <- args[3]; dir.create(out,recursive=TRUE,showWarnings=FALSE)
cells <- read.csv(args[2],colClasses='character',check.names=FALSE)
if (any(cells$cell_type=='HT29') || anyDuplicated(cells$cell_id)) stop('Reserved or duplicate selected cells')
cat('ROUND21 RAW loading pinned sparse RDS\n'); flush.console()
jsonlite::write_json(list(stage='loading RDS',cells=0,total=nrow(cells)),file.path(out,'progress.json'),auto_unbox=TRUE)
obj <- readRDS(args[1]); if (!inherits(obj,'Seurat')) stop('Expected Seurat')
md <- obj[[]]; ix <- match(cells$cell_id,rownames(md)); if (anyNA(ix)) stop('Cell metadata mismatch')
for (field in c('sample','Batch_info','orig.ident','sample_ID','gene','guide')) {
  if (!identical(as.character(md[[field]][ix]),cells[[field]])) stop(paste('Source metadata changed:',field))
}
assay <- obj[['RNA']]
if (inherits(assay,'Assay5')) {
  layers <- SeuratObject::Layers(assay)
  if (sum(grepl('^counts($|\\.)',layers))!=1 || !'counts' %in% layers) stop('Exactly one counts layer required')
  counts <- SeuratObject::LayerData(assay,layer='counts')
} else counts <- methods::slot(assay,'counts')
if (!inherits(counts,'sparseMatrix') || anyDuplicated(colnames(counts))) stop('Expected unique sparse RNA counts')
cols <- match(cells$cell_id,colnames(counts)); if (anyNA(cols)) stop('Missing RNA cells')
library <- numeric(nrow(cells)); detected <- numeric(nrow(cells)); chunk <- 2048L
for (start in seq.int(1,nrow(cells),by=chunk)) {
  ids <- seq.int(start,min(start+chunk-1,nrow(cells))); x <- counts[,cols[ids],drop=FALSE]
  if (any(!is.finite(x@x)) || any(x@x<0) || any(abs(x@x-round(x@x))>1e-5)) stop('Invalid full-library counts')
  library[ids] <- Matrix::colSums(x); detected[ids] <- Matrix::colSums(x>0)
  cat(sprintf('ROUND21 RAW library cells=%d/%d\n',max(ids),nrow(cells))); flush.console()
  jsonlite::write_json(list(stage='library QC',cells=max(ids),total=nrow(cells)),file.path(out,'progress.json'),auto_unbox=TRUE)
}
cells$library_size <- library; cells$detected_genes <- detected
con <- gzfile(file.path(out,'cell_library_qc.csv.gz'),'wt'); write.csv(cells,con,row.names=FALSE); close(con)
jsonlite::write_json(list(state='RAW_LIBRARY_MEASURED',selected_cells=nrow(cells),RNA_gene_count=nrow(counts),
  HT29_columns_selected=FALSE,expression_exported=FALSE,normalization='diagnostic full-library counts only; no changed prepared data'),
  file.path(out,'raw_measurement.json'),pretty=TRUE,auto_unbox=TRUE)
