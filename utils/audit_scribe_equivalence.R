# Docker regression: run inside the existing SCRIBE image with the entrypoint mounted.
suppressPackageStartupMessages(library(monocle))
suppressPackageStartupMessages(library(Scribe))
load_functions <- function(path) {
  env <- new.env(parent=globalenv())
  for (expr in parse(path)) {
    if (is.call(expr) && identical(expr[[1]],as.name('<-')) && is.call(expr[[3]]) && identical(expr[[3]][[1]],as.name('function'))) eval(expr,env)
  }
  env
}
args <- commandArgs(trailingOnly = TRUE)
new <- load_functions(if (length(args)) args[1] else '/runScribe.R')
set.seed(17)
g <- 8L; cells <- 80L
m <- matrix(runif(g*cells)+1,nrow=g,dimnames=list(paste0('g',seq_len(g)),paste0('c',seq_len(cells))))
cd <- new('AnnotatedDataFrame',data=data.frame(Pseudotime=seq_len(cells),State=1,row.names=colnames(m)))
gd <- new('AnnotatedDataFrame',data=data.frame(gene_short_name=rownames(m),row.names=rownames(m)))
cds <- newCellDataSet(as(m,'sparseMatrix'),phenoData=cd,featureData=gd,expressionFamily=uninormal());sizeFactors(cds)<-1
pairs <- new$build_pair_graph(g)
oracle <- function(result, delays) {
  for (i in seq_len(g)) for (j in seq_len(g)) {
    if (i != j) result$max_rdi_delays[i,j] <- as.integer(delays[which.max(result$RDI[i,j+(seq_along(delays)-1L)*g])])
  }
  result
}
for (delays in list(1,1:3,c(3,1,5))) for (uniform in c(FALSE,TRUE)) {
  reference <- oracle(calculate_rdi(cds,delays=delays,method=2,uniformalize=uniform,log=FALSE),delays)
  conditioned <- calculate_conditioned_rdi(cds,rdi_list=reference,uniformalize=uniform,log=FALSE)
  for (workers in c(1L,2L,4L)) {
    result <- new$calculate_rdi_by_pairs(cds,delays,uniform,FALSE,workers,pairs)
    stopifnot(isTRUE(all.equal(reference,result,tolerance=0)))
    actual <- new$calculate_conditioned_rdi_by_pairs(cds,result,uniform,FALSE,workers,pairs)
    stopifnot(isTRUE(all.equal(conditioned,actual,tolerance=0)))
  }
  cat('delays=',paste(delays,collapse=','),' uniform=',uniform,' RDI and conditioned match independent delay oracle across 1/2/4 workers\n',sep='')
}
