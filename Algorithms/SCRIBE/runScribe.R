suppressPackageStartupMessages(library(monocle, warn.conflicts = FALSE , quietly = TRUE))
suppressPackageStartupMessages(library(Scribe, warn.conflicts = FALSE, quietly = TRUE))
suppressPackageStartupMessages(library (optparse, warn.conflicts = FALSE, quietly = TRUE))
suppressPackageStartupMessages(library (igraph, warn.conflicts = FALSE, quietly = TRUE))

cal_ncenter <- function(ncells, ncells_limit = 100){
  round(2 * ncells_limit * log(ncells)/ (log(ncells) + log(ncells_limit)))
}

option_list <- list (
              make_option(c("-e","--expressionFile"), type = 'character',
              help= "Path to comma separated file containing gene-by-cell matrix with
              cell names as the first row and gene names as 
              the first column. Required if -n flag is not used."),
              
              make_option(c("-c","--cellFile"), type = 'character',
              help= "Path to comma separated file containing data on cells.
              First column is Cell name (without a header). Subsequent columns
              contain information on cells (experiment, time, etc).
              Required if -n flag is not used."),
              
              make_option(c("-g","--geneFile"), type = 'character',
              help= "Path to comma separated file containing data on genes.
              First column is gene name (without a header). Subsequent columns
              contain information on genes such as gene_short_name (required). 
              Required if -n flag is not used."),
              
              make_option(c("-n","--newCellDataSet"), type = 'character',
              help= "Path to .RDS file containing an object of type
              newCellDataSet."),
              
              make_option(c("-l","--lowerDetectionLimit"), default = 0.0, type = 'double',
              help= "Single float value to pass as an argument for newCellDataSet function.
              default = %default ."),
              
              make_option(c("-x","--expressionFamily"), default = 'uninormal', type = 'character',
              help= "VGAM family function name (without parantheses) to be used for 
              expression response variables. 
              See more here: http://cole-trapnell-lab.github.io/monocle-release/docs
              default = %default ."),
              
              make_option(c("-m","--method"), default = 'ucRDI', type = 'character',
              help= "Method name for Scribe. Can be any one of 'RDI', 'cRDI', 'uRDI', or 'ucRDI'.
              default = %default ."),
              
              make_option(c("-d","--delay"), default = '1', type = 'character',
              help= "Comma separated list of delay values for Scribe. Maximum delay
              value should be the total number of cells in the dataset. default = %default ."),

              make_option(c("-w","--workers"), default = 1, type = 'integer',
              help= "Independent worker processes used to evaluate directed gene-pair
              chunks. Results are merged before post-processing. default = %default ."),
              
              make_option(c("--log"), action = 'store_true', default = FALSE, type = 'character',
              help= "Log transform expression values. default = %default ."),
              
              make_option(c("-o","--outPrefix"), , type = 'character',
              help= "Path to write output files. Required."),
              
              make_option(c("","--outFile"), , type = 'character',
              help= "outFile name to write the output ranked edges. Required."),
    
              make_option(c("-i","--ignorePT"), action = 'store_true', default = FALSE, 
              type = 'character',
              help= "Ignores pseudotime computed using monocle and uses experiment time.")
              )

parser <- OptionParser(option_list = option_list)
arguments <- parse_args(parser, positional_arguments = FALSE)
print(arguments)


if (length(arguments$outPrefix) == 0){
 stop('outPrefix is required.
      Run Rscript runScribe.R -h for more details.') 
}

# if the new cell dataset file is not provided create one
if (length(arguments$newCellDataSet) == 0){
  
if (length(arguments$expressionFile) == 0){
    stop("Please enter the path to expression file using the -e flag. 
  Run Rscript runScribe.R -h for more details.")
  }
  
if (length(arguments$cellFile) == 0){
    stop("Please enter the path to cell file using the -c flag.
       Run Rscript runScribe.R -h for more details.")
  }
  
  if (length(arguments$geneFile) == 0){
    stop("Please enter the path to gene file using the -g flag.
       Run Rscript runScribe.R -h for more details.")
  }
  
  # Read data
  # Set check.names to False to avoid R adding an 'X' to the beginning of columns that start with an integer
  exprMatrix <- read.delim(arguments$expressionFile, row.names = 1, sep = ',', check.names=FALSE)
  cellData <- read.delim(arguments$cellFile, row.names = 1, sep = ',')
  geneData <- read.delim(arguments$geneFile, row.names = 1, sep = ',')
  cd <- new("AnnotatedDataFrame", data = cellData)
  gd <- new("AnnotatedDataFrame", data = geneData)

# Use uninormal if it is simulated data
if (arguments$expressionFamily == 'uninormal'){
  cat("Using uninormal() as expression family.\n")
  CDS <- newCellDataSet(as(as.matrix(exprMatrix), "sparseMatrix"),
                       phenoData = cd,
                       featureData = gd,
                       lowerDetectionLimit = arguments$lowerDetectionLimit,
                       expressionFamily = uninormal())

sizeFactors(CDS) <- 1 # Same as the one in neuronal_sim_cCDS
} else{
  # For scRNA-Seq data (counts, RPKM/FPKM)
  cat("Using negbinomial.size() as expression family.\n")
CDS <- newCellDataSet(as(as.matrix(exprMatrix), "sparseMatrix"),
                       phenoData = cd,
                       featureData = gd,
                       lowerDetectionLimit = arguments$lowerDetectionLimit,
                       expressionFamily = negbinomial.size())
  
CDS <- estimateSizeFactors(CDS)
CDS <- estimateDispersions(CDS)
disp_table <- dispersionTable(CDS)
ordering_genes <- disp_table$gene_id
CDS <- setOrderingFilter(CDS, ordering_genes)
}

if (arguments$ignorePT == TRUE){
  cat("Using experimental time instead of PseudoTime computed using
      monocle.\n")
  CDS$Pseudotime <- CDS$Time
  CDS@phenoData@data$Pseudotime <- CDS@phenoData@data$Time
  CDS@phenoData@data$State <- 1
  CDS$State <- 1
  
}else{
  
cat("Computing pseudotime.\n")
CDS <- reduceDimension(CDS, norm_method ="none")
CDS <- orderCells(CDS)

saveRDS(CDS, file= paste0(arguments$outPrefix,'dataset.RDS'))
write.csv(CDS@phenoData@data, file= paste0(arguments$outPrefix,'PseudoTime.csv'), quote = FALSE)
}
}

# If newcelldataset RDS is available already
if (length(arguments$newCellDataSet) != 0){
  CDS <- readRDS(arguments$newCellDataSet)
}

### Run Scribe

cat("Computing",arguments$method,"\n")


delay <- as.numeric(strsplit(arguments$delay, ",")[[1]])

build_pair_graph <- function(n_genes) {
  gene_ids <- seq.int(0L, n_genes - 1L)
  pair_graph <- as.matrix(expand.grid(gene_ids, gene_ids))
  pair_graph <- pair_graph[pair_graph[, 1] != pair_graph[, 2], , drop = FALSE]
  storage.mode(pair_graph) <- "integer"
  pair_graph
}

split_pair_graph <- function(pair_graph, workers) {
  worker_count <- max(1L, min(as.integer(workers), nrow(pair_graph)))
  split(
    seq_len(nrow(pair_graph)),
    rep(seq_len(worker_count), length.out = nrow(pair_graph))
  )
}

merge_rdi_chunks <- function(chunk_results, pair_graph, pair_chunks, delays) {
  merged <- chunk_results[[1]]
  merged$RDI[,] <- 0
  merged$max_rdi_value[,] <- 0
  merged$max_rdi_delays[,] <- 0
  n_genes <- nrow(merged$max_rdi_value)

  for (chunk_index in seq_along(pair_chunks)) {
    pair_rows <- pair_chunks[[chunk_index]]
    current_pairs <- pair_graph[pair_rows, , drop = FALSE]
    source_ids <- current_pairs[, 1] + 1L
    target_ids <- current_pairs[, 2] + 1L
    matrix_indices <- cbind(source_ids, target_ids)
    current_result <- chunk_results[[chunk_index]]

    merged$max_rdi_value[matrix_indices] <-
      current_result$max_rdi_value[matrix_indices]
    merged$max_rdi_delays[matrix_indices] <-
      current_result$max_rdi_delays[matrix_indices]

    for (delay_index in seq_along(delays)) {
      rdi_indices <- cbind(
        source_ids,
        target_ids + (delay_index - 1L) * n_genes
      )
      merged$RDI[rdi_indices] <- current_result$RDI[rdi_indices]
    }
  }
  merged
}

calculate_rdi_by_pairs <- function(CDS, delays, uniformalize, log_values,
                                   workers, pair_graph) {
  if (workers <= 1L) {
    return(calculate_rdi(
      CDS,
      delays = delays,
      method = 2,
      uniformalize = uniformalize,
      log = log_values
    ))
  }

  pair_chunks <- split_pair_graph(pair_graph, workers)
  chunk_results <- parallel::mclapply(
    pair_chunks,
    function(pair_rows) {
      calculate_rdi(
        CDS,
        delays = delays,
        super_graph = pair_graph[pair_rows, , drop = FALSE],
        method = 2,
        uniformalize = uniformalize,
        log = log_values
      )
    },
    mc.cores = length(pair_chunks),
    mc.preschedule = TRUE,
    mc.set.seed = FALSE
  )
  failed_chunks <- vapply(chunk_results, inherits, logical(1), "try-error")
  if (any(failed_chunks)) {
    stop(
      "SCRIBE RDI worker failed: ",
      paste(as.character(chunk_results[failed_chunks]), collapse = "; ")
    )
  }
  merge_rdi_chunks(
    chunk_results,
    pair_graph,
    pair_chunks,
    delays
  )
}

calculate_conditioned_chunk <- function(CDS, pair_graph, rdi_list,
                                        uniformalize, log_values) {
  size_factors <- sizeFactors(CDS)
  size_factors[is.na(size_factors)] <- 1
  ordered_cells <- order(pData(CDS)$Pseudotime)
  genes_data <- t(as.matrix(exprs(CDS)[, ordered_cells])) /
    size_factors[ordered_cells]
  if (log_values) {
    genes_data <- log(genes_data + 1)
  }
  if (!all(is.finite(genes_data))) {
    stop("your data includes non finite values!")
  }

  conditioned <- calculate_conditioned_rdi_cpp_wrap(
    genes_data,
    pair_graph,
    rdi_list$max_rdi_value,
    rdi_list$max_rdi_delays,
    1L,
    uniformalize
  )
  dimnames(conditioned) <- list(colnames(genes_data), colnames(genes_data))
  conditioned
}

calculate_conditioned_rdi_by_pairs <- function(CDS, rdi_list, uniformalize,
                                               log_values, workers,
                                               pair_graph) {
  if (workers <= 1L) {
    return(calculate_conditioned_rdi(
      CDS,
      rdi_list = rdi_list,
      uniformalize = uniformalize,
      log = log_values
    ))
  }

  pair_chunks <- split_pair_graph(pair_graph, workers)
  chunk_results <- parallel::mclapply(
    pair_chunks,
    function(pair_rows) {
      calculate_conditioned_chunk(
        CDS,
        pair_graph[pair_rows, , drop = FALSE],
        rdi_list,
        uniformalize,
        log_values
      )
    },
    mc.cores = length(pair_chunks),
    mc.preschedule = TRUE,
    mc.set.seed = FALSE
  )
  failed_chunks <- vapply(chunk_results, inherits, logical(1), "try-error")
  if (any(failed_chunks)) {
    stop(
      "SCRIBE conditioned-RDI worker failed: ",
      paste(as.character(chunk_results[failed_chunks]), collapse = "; ")
    )
  }

  merged <- chunk_results[[1]]
  merged[,] <- 0
  for (chunk_index in seq_along(pair_chunks)) {
    current_pairs <- pair_graph[pair_chunks[[chunk_index]], , drop = FALSE]
    matrix_indices <- cbind(
      current_pairs[, 1] + 1L,
      current_pairs[, 2] + 1L
    )
    merged[matrix_indices] <- chunk_results[[chunk_index]][matrix_indices]
  }
  merged
}

n_genes <- nrow(exprs(CDS))
pair_count <- n_genes * (n_genes - 1L)
requested_workers <- max(1L, as.integer(arguments$workers))
pair_workers <- max(1L, min(requested_workers, pair_count))
pair_graph <- if (pair_workers > 1L) build_pair_graph(n_genes) else NULL
cat("Using", pair_workers, "worker process(es) for", pair_count,
    "directed gene pairs.\n")

if (arguments$method == 'uRDI'){
  net <- calculate_rdi_by_pairs(CDS, delay, TRUE, arguments$log,
                                pair_workers, pair_graph)
  netOut <- net$max_rdi_value
  # computes CLR if we use uRDI
  # TODO: Make this an optional
  netOut <- clr(netOut)
} else if (arguments$method == 'ucRDI'){
  net <- calculate_rdi_by_pairs(CDS, delay, TRUE, arguments$log,
                                pair_workers, pair_graph)
  netOut <- calculate_conditioned_rdi_by_pairs(
    CDS, net, TRUE, arguments$log, pair_workers, pair_graph
  )
} else if (arguments$method == 'RDI'){
  net <- calculate_rdi_by_pairs(CDS, delay, FALSE, arguments$log,
                                pair_workers, pair_graph)
  netOut <- net$max_rdi_value
  # computes CLR if we use RDI
  # TODO: Make this optional
  netOut <- clr(netOut)
} else if (arguments$method == 'cRDI'){
  net <- calculate_rdi_by_pairs(CDS, delay, FALSE, arguments$log,
                                pair_workers, pair_graph)
  netOut <- calculate_conditioned_rdi_by_pairs(
    CDS, net, FALSE, arguments$log, pair_workers, pair_graph
  )
} else{
  stop("Method must be one of RDI, cRDI, uRDI, or ucRDI. 
       Run Rscript runScribe.R -h for more details.")
}
outGraph <- graph_from_adjacency_matrix(netOut, mode = 'directed', weighted=T)
write.graph(outGraph, paste0(arguments$outPrefix,arguments$outFile),"ncol")
cat("Done.\n")
