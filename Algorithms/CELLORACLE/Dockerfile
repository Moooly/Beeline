# CellOracle GRN inference for BEELINE.
#
# Uses CellOracle's built-in promoter / mouse-scATAC base GRNs (loaded via
# co.data.load_*_base_GRN), so the heavy genomic base-GRN-construction stack
# (genomepy / gimmemotifs / pybedtools) is NOT required for this inference path.
#
# NOTE: CellOracle's dependency stack is large and occasionally finicky; this
# Dockerfile is expected to need iteration when first built on the server.
FROM continuumio/miniconda3:24.1.2-0

LABEL maintainer="GRNScope"

USER root
WORKDIR /

# Pin Python for CellOracle compatibility, then install CellOracle (pulls
# scanpy, scikit-learn, numba, python-igraph, anndata, etc.).
RUN conda install -y -n base -c conda-forge python=3.10 && \
    conda clean -afy

RUN pip install --no-cache-dir celloracle scanpy

# Bake the built-in base GRNs into the image so runtime containers need no
# network access. Cache under a fixed, world-readable HOME.
ENV HOME=/root
COPY prefetch_base_grns.py /
RUN python /prefetch_base_grns.py && chmod -R a+rX /root/.celloracle 2>/dev/null || true

COPY runCellOracle.py /

RUN apt-get update && apt-get install -y time && rm -rf /var/lib/apt/lists/*
