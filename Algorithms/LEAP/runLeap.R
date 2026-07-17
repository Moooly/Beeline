library(LEAP)

args <- commandArgs(trailingOnly = T)
inFile <- args[1]
maxLag <- as.numeric(args[2])
outFile <-  args[3]
topK <- if (length(args) >= 4) as.integer(args[4]) else 0L

# input expression data
inputExpr <- read.table(inFile, sep=",", header = 1, row.names = 1)
geneNames <- rownames(inputExpr)
rownames(inputExpr) <- c()
# Run LEAP's compute Max. Absolute Correlation
# MAC_cutoff is set to zero to get a score for all TFs
# max_lag_prop is set to the max. recommended value from the paper's supplementary file
# Link to paper: https://academic.oup.com/bioinformatics/article/33/5/764/2557687

MAC_results = MAC_counter(data = inputExpr, max_lag_prop=maxLag, MAC_cutoff = 0, 
                          file_name = "temp", lag_matrix = FALSE, symmetric = FALSE)

# Write output to a file
Gene1 <- geneNames[MAC_results[,'Row gene index']]
Gene2 <- geneNames[MAC_results[,'Column gene index']]
Score <- MAC_results[,'Correlation']
outDF <- data.frame(Gene1, Gene2, Score)
if (topK > 0) {
    outDF$.sequence <- seq_len(nrow(outDF))
    groups <- split(outDF, outDF$Gene2, drop = TRUE)
    groups <- lapply(groups, function(group) {
        ordered <- group[order(-abs(group$Score), group$.sequence), ]
        ordered[seq_len(min(topK, nrow(ordered))), ]
    })
    outDF <- do.call(rbind, groups)
    outDF <- outDF[order(outDF$.sequence), c("Gene1", "Gene2", "Score")]
}
write.table(outDF, outFile, sep = "\t", quote = FALSE, row.names = FALSE)
