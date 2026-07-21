# Slingshot-based pseudotime estimator for GRNScope / BEELINE.
#
# Builds a small R image with Slingshot (Bioconductor) + optparse (CRAN) and the
# estimate_pseudotime.R entrypoint. Given an ExpressionData.csv it writes a
# BEELINE-compatible PseudoTime.csv, unlocking the trajectory-based GRN methods
# for datasets uploaded without their own pseudotime.
#
# Build:  docker build -t grnbeeline/slingshot:0.1.0 .
# Run:    docker run --rm -v "$PWD:/data" grnbeeline/slingshot:0.1.0 \
#           --expression /data/ExpressionData.csv --output /data/PseudoTime.csv
#
# The Bioconductor base pins R + BiocManager so slingshot installs reliably.
FROM bioconductor/bioconductor_docker:RELEASE_3_18

# CRAN dependency for CLI parsing.
RUN R --no-save -e "install.packages('optparse', repos='https://cloud.r-project.org')"

# Slingshot from Bioconductor. DelayedMatrixStats is only *suggested* by
# slingshot's dependency chain and is loaded lazily at run time (inside
# slingshot()), so it must be installed explicitly or the run fails with
# "there is no package called 'DelayedMatrixStats'".
RUN R --no-save -e "BiocManager::install(c('slingshot', 'SingleCellExperiment', 'DelayedMatrixStats'), update = FALSE, ask = FALSE)"

# Fail the build early if anything (including the lazily-loaded namespaces
# slingshot() reaches for) is missing.
RUN R --no-save -e "library(slingshot); library(optparse); loadNamespace('DelayedMatrixStats'); cat('deps-ok\n')"

WORKDIR /app
COPY estimate_pseudotime.R /app/estimate_pseudotime.R

ENTRYPOINT ["Rscript", "/app/estimate_pseudotime.R"]
