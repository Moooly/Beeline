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

# Slingshot (and SingleCellExperiment, pulled in as a dependency) from Bioconductor.
RUN R --no-save -e "BiocManager::install(c('slingshot', 'SingleCellExperiment'), update = FALSE, ask = FALSE)"

# Fail the build early if the packages didn't install.
RUN R --no-save -e "library(slingshot); library(optparse); cat('deps-ok\n')"

WORKDIR /app
COPY estimate_pseudotime.R /app/estimate_pseudotime.R

ENTRYPOINT ["Rscript", "/app/estimate_pseudotime.R"]
