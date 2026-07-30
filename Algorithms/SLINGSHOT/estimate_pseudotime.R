#!/usr/bin/env Rscript

# Estimate a pseudotime trajectory for a scRNA-seq expression matrix using the
# same pipeline BEELINE's preprocessing uses: state-aware expression preparation
# -> PCA -> cluster -> Slingshot. Produces a BEELINE-compatible PseudoTime.csv
# so datasets that arrive without pseudotime can still run the trajectory-based
# GRN methods.
#
# See README.md for the full interface. Summary:
#   Rscript estimate_pseudotime.R \
#     --expression ExpressionData.csv \
#     [--clusters ClusterLabels.csv] \
#     [--startCluster <label>] \
#     [--matrixState raw|normalized|log_normalized] \
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
  make_option(c("--matrixState"), type = "character", default = "raw",
              help = paste0(
                "Input expression state: raw, normalized, or log_normalized. ",
                "GRNScope passes log_normalized after applying its shared preprocessing ",
                "contract [default %default]."
              )),
  make_option(c("-o", "--output"), type = "character", default = "PseudoTime.csv",
              help = "Output PseudoTime.csv path [default %default]."),
  make_option(c("--embeddingOutput"), type = "character", default = NULL,
              help = "Optional CSV path for the PCA coordinates used by Slingshot."),
  make_option(c("--curvesOutput"), type = "character", default = NULL,
              help = "Optional CSV path for the fitted Slingshot principal curves."),
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

# --- 2. Respect the declared matrix state and drop zero-variance genes --------
matrix_state <- tolower(trimws(opt$matrixState))
if (!(matrix_state %in% c("raw", "normalized", "log_normalized"))) {
  fail("--matrixState must be raw, normalized, or log_normalized.")
}
if (any(!is.finite(cell_by_gene))) {
  fail("Expression matrix contains a non-finite value.")
}
if (matrix_state %in% c("raw", "normalized") && any(cell_by_gene < 0)) {
  fail("Raw and normalized expression matrices cannot contain negative values.")
}

analysis_expression <- cell_by_gene
applied_operations <- character(0)
if (matrix_state == "raw") {
  cell_totals <- rowSums(analysis_expression)
  positive_totals <- cell_totals > 0
  analysis_expression[positive_totals, ] <- (
    analysis_expression[positive_totals, , drop = FALSE] *
      (10000 / cell_totals[positive_totals])
  )
  applied_operations <- c(applied_operations, "normalize_total")
}
if (matrix_state %in% c("raw", "normalized")) {
  analysis_expression <- log1p(analysis_expression)
  applied_operations <- c(applied_operations, "log1p")
}
cat(sprintf(
  "Matrix state: %s; applied operations: %s\n",
  matrix_state,
  if (length(applied_operations)) paste(applied_operations, collapse = ", ") else "none"
))

gene_variance <- apply(analysis_expression, 2, stats::var)
analysis_expression <- analysis_expression[, gene_variance > 0, drop = FALSE]
if (ncol(analysis_expression) < 2) {
  fail("No variable genes remain after filtering; cannot build a trajectory.")
}

# --- 3. PCA embedding --------------------------------------------------------
n_pcs <- min(opt$nPCs, ncol(analysis_expression), nrow(analysis_expression) - 1)
if (n_pcs < 2) {
  fail("Not enough dimensions for a PCA embedding.")
}
cat(sprintf("Computing PCA (%d components)\n", n_pcs))
pca <- stats::prcomp(analysis_expression, center = TRUE, scale. = FALSE)
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

lineage_names <- paste0("PseudoTime", seq_len(ncol(pseudotime)))
colnames(pseudotime) <- lineage_names
rownames(pseudotime) <- cell_names
cat(sprintf("Recovered %d lineage(s) across %d cells\n", ncol(pseudotime), nrow(pseudotime)))

# Preserve the coordinate system and the actual principal curves used by
# Slingshot. GRNScope can then render the fitted trajectory instead of
# reconstructing a visual guide from pseudotime-bin centroids.
if (!is.null(opt$embeddingOutput)) {
  embedding_frame <- data.frame(
    cell = cell_names,
    x = reduced[, 1],
    y = reduced[, 2],
    check.names = FALSE
  )
  write.csv(
    embedding_frame,
    file = opt$embeddingOutput,
    quote = FALSE,
    row.names = FALSE
  )
  cat(sprintf("Wrote %s\n", opt$embeddingOutput))
}

if (!is.null(opt$curvesOutput)) {
  fitted_curves <- slingCurves(slingshot_fit)
  curve_frames <- lapply(seq_along(fitted_curves), function(lineage_index) {
    curve <- fitted_curves[[lineage_index]]
    curve_order <- curve$ord
    curve_coordinates <- curve$s[curve_order, , drop = FALSE]
    curve_pseudotime <- curve$lambda[curve_order]
    finite <- (
      is.finite(curve_coordinates[, 1]) &
      is.finite(curve_coordinates[, 2]) &
      is.finite(curve_pseudotime)
    )
    curve_coordinates <- curve_coordinates[finite, , drop = FALSE]
    curve_pseudotime <- curve_pseudotime[finite]
    data.frame(
      lineage = lineage_names[lineage_index],
      point_order = seq_len(nrow(curve_coordinates)),
      x = curve_coordinates[, 1],
      y = curve_coordinates[, 2],
      pseudotime = curve_pseudotime,
      check.names = FALSE
    )
  })
  curve_frames <- Filter(function(frame) nrow(frame) > 0, curve_frames)
  if (length(curve_frames)) {
    curve_frame <- do.call(rbind, curve_frames)
  } else {
    curve_frame <- data.frame(
      lineage = character(0),
      point_order = integer(0),
      x = numeric(0),
      y = numeric(0),
      pseudotime = numeric(0),
      check.names = FALSE
    )
  }
  write.csv(
    curve_frame,
    file = opt$curvesOutput,
    quote = FALSE,
    row.names = FALSE
  )
  cat(sprintf("Wrote %s\n", opt$curvesOutput))
}

# --- 7. Write PseudoTime.csv -------------------------------------------------
# Standard index+lineage layout: an empty corner cell heads the cell-name
# column, then one column per Slingshot lineage. `col.names = NA` writes that
# empty corner, so the header lines up with the data rows in any viewer and
# pandas still reads it cleanly with index_col=0:
#     ,PseudoTime1,PseudoTime2
#     CELL_A,114.76,NA
write.table(
  pseudotime,
  file = opt$output,
  sep = ",",
  quote = FALSE,
  col.names = NA,
  row.names = TRUE,
  na = "NA"
)
cat(sprintf("Wrote %s\n", opt$output))
