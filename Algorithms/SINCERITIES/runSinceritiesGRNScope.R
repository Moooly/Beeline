library(kSamples)
library(glmnet)
library(ppcor)

args <- commandArgs(trailingOnly = TRUE)
inFile <- args[1]
outFile <- args[2]
topK <- if (length(args) >= 3) as.integer(args[3]) else 0L

# These functions are supplied by the original SINCERITIES image. The
# inference call and parameters are intentionally identical to MAIN.R.
uploading <- dget("SINCERITIES functions/uploading.R")
DATA <- uploading(inFile)
SINCERITITES <- dget("SINCERITIES functions/SINCERITIES.R")
result <- tryCatch(
    SINCERITITES(DATA, distance = 1, method = 1, noDIAG = 0, SIGN = 1),
    error = function(error) {
        if (grepl("y is constant", conditionMessage(error), fixed = TRUE)) {
            stop(paste0("SINCERITIES cannot fit a constant distribution-change response. ",
                        "Try more time bins or a trajectory with more varied expression."), call. = FALSE)
        }
        stop(error)
    }
)

peak <- max(result$adj_matrix)
if (is.finite(peak) && peak <= 0) peak <- max(abs(result$adj_matrix))
if (!is.finite(peak)) stop("SINCERITIES returned nonfinite adjacency scores.")
adj_matrix <- if (peak == 0) result$adj_matrix else result$adj_matrix / peak
final_ranked_predictions <- dget(
    "SINCERITIES functions/final_ranked_predictions.R"
)
table <- final_ranked_predictions(
    adj_matrix,
    DATA$genes,
    SIGN = 1,
    saveFile = FALSE
)

table <- table[table$SourceGENES != table$TargetGENES, ]
if (topK > 0 && nrow(table) > 0) {
    # GRNScope's parser corrects SINCERITIES' reversed orientation, making
    # SourceGENES the downstream target. The table is already ordered by
    # decreasing Interaction, so the first K rows per source are exact.
    table$.sequence <- seq_len(nrow(table))
    groups <- split(table, table$SourceGENES, drop = TRUE)
    groups <- lapply(groups, function(group) {
        group <- group[order(-ifelse(tolower(group$Edges) == "no regulation", 0, abs(group$Interaction)), group$TargetGENES, method = "radix"), ]
        group[seq_len(min(topK, nrow(group))), ]
    })
    table <- do.call(rbind, groups)
    table <- table[
        order(table$.sequence),
        c("SourceGENES", "TargetGENES", "Interaction", "Edges")
    ]
}

write.csv(table, file = outFile, row.names = FALSE, quote = FALSE)
