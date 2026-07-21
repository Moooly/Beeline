#!/usr/bin/env Rscript

# Estimate a pseudotime trajectory for a scRNA-seq expression matrix using the
# same pipeline BEELINE's preprocessing uses: log-transform -> PCA -> cluster
# -> Slingshot. Produces a BEELINE-compatible PseudoTime.csv so datasets that
# arrive without pseudotime can still run the trajectory-based GRN methods.
#
# See README.md for the full interface. Summary:
#   Rscript estimate_pseudotime.R \
#     --expression ExpressionData.csv \
#     [--clusters ClusterLabels.csv] \
#     [--startCluster <label>] \
#     [--nPCs 20] [--nClusters 10] [--seed 1] \
#     --output PseudoTime.csv

suppressPackageStartupMessages({
  library(optparse)
  library(slingshot)
})

option_list <- list(
  make_option(c("-e", "--expression"), type = "character", default = NULL,
              help = "Path to ExpressionData.csv (genes x cells). Required."),
  make_option(c("-c", "--clusters"), type = "character", default = NULL,
              help = "Optional ClusterLabels.csv (cell name in first column, cluster label in the next)."),
  make_option(c("-s", "--startCluster"), type = "character", default = NULL,
              help = "Optional starting cluster label used as the trajectory root."),
  make_option(c("-o", "--output"), type = "character", default = "PseudoTime.csv",
              help = "Output PseudoTime.csv path [default %default]."),
  make_option(c("--nPCs"), type = "integer", default = 20,
              help = "Number of principal components used for the embedding [default %default]."),
  make_option(c("--nClusters"), type = "integer", default = 10,
              help = "k for k-means clustering when no cluster file is given [default %default]."),
  make_option(c("--seed"), type = "integer", default = 1,
              help = "Random seed for reproducibility [default %default].")
)

opt <- parse_args(OptionParser(option_list = option_list))

fail <- function(message) {
  cat(sprintf("ERROR: %s\n", message), file = stderr())
  quit(status = 1)
}

if (is.null(opt$expression)) {
  fail("--expression is required.")
}
if (!file.exists(opt$expression)) {
  fail(sprintf("Expression file not found: %s", opt$expression))
}

set.seed(opt$seed)

# --- 1. Read expression (genes x cells) and orient to cells x genes ----------
cat(sprintf("Reading expression matrix: %s\n", opt$expression))
expression <- read.csv(opt$expression, header = TRUE, row.names = 1, check.names = FALSE)
expression <- as.matrix(expression)
if (nrow(expression) < 2 || ncol(expression) < 3) {
  fail("Expression matrix is too small to build a trajectory.")
}

# Analysis works on cells x genes.
cell_by_gene <- t(expression)
cell_names <- rownames(cell_by_gene)
cat(sprintf("  %d cells x %d genes\n", nrow(cell_by_gene), ncol(cell_by_gene)))

# --- 2. Log-transform and drop zero-variance genes ---------------------------
log_expression <- log2(cell_by_gene + 1)
gene_variance <- apply(log_expression, 2, stats::var)
log_expression <- log_expression[, gene_variance > 0, drop = FALSE]
if (ncol(log_expression) < 2) {
  fail("No variable genes remain after filtering; cannot build a trajectory.")
}

# --- 3. PCA embedding --------------------------------------------------------
n_pcs <- min(opt$nPCs, ncol(log_expression), nrow(log_expression) - 1)
if (n_pcs < 2) {
  fail("Not enough dimensions for a PCA embedding.")
}
cat(sprintf("Computing PCA (%d components)\n", n_pcs))
pca <- stats::prcomp(log_expression, center = TRUE, scale. = FALSE)
reduced <- pca$x[, seq_len(n_pcs), drop = FALSE]

# --- 4. Cluster labels (provided, or k-means) --------------------------------
if (!is.null(opt$clusters)) {
  if (!file.exists(opt$clusters)) {
    fail(sprintf("Cluster file not found: %s", opt$clusters))
  }
  cat(sprintf("Reading cluster labels: %s\n", opt$clusters))
  # Expected format (matches GRNScope): two columns `cell_id,cluster`, with an
  # OPTIONAL header row. Read everything as character and auto-detect the header
  # so both headered and header-less files work, and numeric cluster ids
  # ("0", "1", ...) aren't coerced to numbers.
  cluster_frame <- read.csv(opt$clusters, header = FALSE, colClasses = "character",
                            check.names = FALSE, stringsAsFactors = FALSE)
  if (ncol(cluster_frame) < 2) {
    fail("Cluster file must have two columns: cell name and cluster label.")
  }
  # Drop a header row if its first value isn't one of the expression cells.
  first_value <- trimws(as.character(cluster_frame[[1]][1]))
  if (!(first_value %in% cell_names)) {
    cluster_frame <- cluster_frame[-1, , drop = FALSE]
  }
  cluster_labels <- trimws(as.character(cluster_frame[[2]]))
  names(cluster_labels) <- trimws(as.character(cluster_frame[[1]]))
  cluster_labels <- cluster_labels[cell_names]
  if (any(is.na(cluster_labels))) {
    fail("Some cells in the expression matrix have no matching cluster label.")
  }
} else {
  k <- min(opt$nClusters, nrow(reduced) - 1)
  k <- max(k, 2)
  cat(sprintf("No cluster file provided; running k-means with k=%d\n", k))
  kmeans_fit <- stats::kmeans(reduced, centers = k, nstart = 25, iter.max = 100)
  cluster_labels <- as.character(kmeans_fit$cluster)
  names(cluster_labels) <- cell_names
}

if (length(unique(cluster_labels)) < 2) {
  fail(paste0(
    "A trajectory needs at least 2 cell clusters, but only one was found. ",
    "Provide cluster labels or increase --nClusters."
  ))
}

# --- 5. Validate the optional start cluster ----------------------------------
start_cluster <- opt$startCluster
if (!is.null(start_cluster)) {
  if (!(start_cluster %in% cluster_labels)) {
    fail(sprintf("Start cluster '%s' was not found among the cluster labels.", start_cluster))
  }
  cat(sprintf("Using start cluster: %s\n", start_cluster))
} else {
  cat("No start cluster given; Slingshot will choose the trajectory root.\n")
}

# --- 6. Slingshot ------------------------------------------------------------
cat("Fitting Slingshot trajectory...\n")
slingshot_fit <- slingshot(
  reduced,
  clusterLabels = cluster_labels,
  start.clus = start_cluster
)
pseudotime <- slingPseudotime(slingshot_fit)  # cells x lineages, NA off-lineage
if (is.null(dim(pseudotime))) {
  pseudotime <- matrix(pseudotime, ncol = 1)
}

colnames(pseudotime) <- paste0("PseudoTime", seq_len(ncol(pseudotime)))
rownames(pseudotime) <- cell_names
cat(sprintf("Recovered %d lineage(s) across %d cells\n", ncol(pseudotime), nrow(pseudotime)))

# --- 7. Write BEELINE-compatible PseudoTime.csv ------------------------------
# BEELINE format: header lists the lineage columns only (no corner cell), each
# data row is `<cell>,<value|NA>,...`. `col.names = TRUE, row.names = TRUE`
# produces exactly that shifted header, which pandas reads with index_col=0.
suppressWarnings(
  write.table(
    pseudotime,
    file = opt$output,
    sep = ",",
    quote = FALSE,
    col.names = TRUE,
    row.names = TRUE,
    na = "NA"
  )
)
cat(sprintf("Wrote %s\n", opt$output))
