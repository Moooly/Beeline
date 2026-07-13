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

# Build tooling for velocyto (a CellOracle dependency) whose wheel build compiles
# Cython extensions and imports numpy at build time.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential time && \
    rm -rf /var/lib/apt/lists/*

# velocyto does not declare numpy as a build dependency and imports it during its
# wheel build, so install numpy + cython first and build velocyto WITHOUT
# isolation against them. Fixes the "No module named 'numpy'" wheel-build failure.
RUN pip install --no-cache-dir "numpy<2" cython
RUN pip install --no-cache-dir --no-build-isolation velocyto==0.17.17

# CellOracle pulls its own compatible scanpy/anndata. Do NOT force a newer scanpy
# here — it would conflict with CellOracle's pandas<=1.5.3 pin.
RUN pip install --no-cache-dir celloracle

# Bake the built-in base GRNs into the image so runtime containers need no
# network access. Cache under a fixed, world-readable HOME.
ENV HOME=/root
COPY prefetch_base_grns.py /
RUN python /prefetch_base_grns.py && chmod -R a+rX /root/.celloracle 2>/dev/null || true

COPY runCellOracle.py /
