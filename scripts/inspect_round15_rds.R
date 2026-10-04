# Metadata and sparse-count QC only. No DE, response aggregation or automatic mapping.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2) stop("Usage: Rscript inspect_round15_rds.R input.rds output_dir")
for (pkg in c("SeuratObject", "Matrix", "jsonlite")) {
  if (!requireNamespace(pkg, quietly=TRUE)) stop(paste("Missing R package",pkg))
}
out <- args[2]; dir.create(out,recursive=TRUE,showWarnings=FALSE)
obj <- readRDS(args[1])
if (!inherits(obj,"Seurat")) stop("Expected a Seurat object; explicit adapter required")
metadata <- obj[[]]
metadata$cell_id <- rownames(metadata)
con <- gzfile(file.path(out,"metadata.csv.gz"),"wt")
write.csv(metadata,con,row.names=FALSE);close(con)
fields <- lapply(names(metadata),function(n) {
  x <- as.character(metadata[[n]]); tab <- sort(table(x,useNA="ifany"),decreasing=TRUE)
  list(name=n,unique=length(unique(x)),missing=sum(is.na(x)),
       values=if (length(tab)<=200) as.list(tab) else NULL)
})
assays <- names(obj@assays); layers <- list()
for (a in assays) {
  assay <- obj[[a]]
  available <- if (inherits(assay,"Assay5")) SeuratObject::Layers(assay) else intersect(c("counts","data","scale.data"),slotNames(assay))
  for (layer in available) {
    if (!grepl("^counts($|\\.)",layer)) next
    x <- if (inherits(assay,"Assay5")) SeuratObject::LayerData(assay,layer=layer) else methods::slot(assay,layer)
    if (!inherits(x,"sparseMatrix")) stop(paste("Counts are not sparse; refusing dense conversion",a,layer))
    values <- x@x
    layers[[paste(a,layer,sep="/")]] <- list(assay=a,layer=layer,genes=nrow(x),cells=ncol(x),
      nonzero=length(values),finite=all(is.finite(values)),nonnegative=all(values>=0),
      integer_counts=all(abs(values-round(values))<1e-5),duplicate_genes=sum(duplicated(rownames(x))),
      duplicate_cells=sum(duplicated(colnames(x))))
    # Gene identifiers and measured coverage are metadata, not response-based feature selection.
    safe <- gsub("[^A-Za-z0-9_-]","_",paste(a,layer,sep="_"))
    write.table(rownames(x),file.path(out,paste0(safe,"_genes.txt")),quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
if (length(layers)==0) stop("No raw sparse counts layer found")
jsonlite::write_json(list(status="METADATA_INSPECTED_NOT_TRAINING_READY",cells=nrow(metadata),
  fields=fields,count_layers=layers,assays=assays,
  reserved_cell_line="HT29",response_analysis=FALSE,
  next_required="Explicit field mapping, modality/guide/condition/control validation and HT29 reservation enforcement"),
  file.path(out,"rds_inspect.json"),pretty=TRUE,auto_unbox=TRUE)
