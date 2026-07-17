"""In-container CellOracle GRN inference for BEELINE.

Reads a genes-by-cells expression matrix, runs CellOracle's prior-informed GRN
inference on the given cells as a single GRN unit (per-cluster scoping is handled
upstream by GRNScope, which passes one cluster's cells at a time), and writes a
ranked directed/signed edge list.

Pipeline (CellOracle standard):
  expression -> AnnData -> Oracle.import_anndata_as_raw_count
  -> import_TF_data(base GRN)
  -> perform_PCA + knn_imputation -> get_links -> filter by p-value -> edge list

The base GRN is a species-specific promoter network (or the mouse scATAC atlas),
loaded from CellOracle's bundled data (baked into the image at build time).
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import pandas as pd
import anndata as ad
import celloracle as co
import scanpy as sc


# Species -> CellOracle built-in promoter base-GRN loader. Keys match the
# GRNScope registry's CELLORACLE_SPECIES_OPTIONS.
PROMOTER_BASE_GRN_LOADERS = {
    "human": co.data.load_human_promoter_base_GRN,
    "mouse": co.data.load_mouse_promoter_base_GRN,
    "rat": co.data.load_rat_promoter_base_GRN,
    "pig": co.data.load_Pig_promoter_base_GRN,
    "chicken": co.data.load_chicken_promoter_base_GRN,
    "zebrafish": co.data.load_zebrafish_promoter_base_GRN,
    "xenopus_tropicalis": co.data.load_xenopus_tropicalis_promoter_base_GRN,
    "drosophila": co.data.load_drosophila_promoter_base_GRN,
    "c_elegans": co.data.load_Celegans_promoter_base_GRN,
    "s_cerevisiae": co.data.load_Scerevisiae_promoter_base_GRN,
}

GRN_UNIT_COLUMN = "grn_unit"


@lru_cache(maxsize=2)
def _load_tf_dictionary_cached(species: str, base_grn: str) -> dict:
    """Load and convert one base GRN per persistent worker/species/mode.

    This mirrors ``Oracle.import_TF_data(TF_info_matrix=...)`` exactly. Passing
    the cached dictionary on warm requests avoids copying and regrouping the
    full promoter matrix each time.
    """
    if base_grn == "mouse_scATAC_atlas":
        if species != "mouse":
            raise ValueError("mouse_scATAC_atlas base GRN is only available for mouse.")
        base_grn_frame = co.data.load_mouse_scATAC_atlas_base_GRN()
    else:
        loader = PROMOTER_BASE_GRN_LOADERS.get(species)
        if loader is None:
            raise ValueError(f"No built-in CellOracle base GRN for species '{species}'.")
        base_grn_frame = loader()

    grouped = (
        base_grn_frame.drop(["peak_id"], axis=1)
        .groupby(by="gene_short_name")
        .sum()
    )
    return dict(
        grouped.apply(lambda column: column[column > 0].index.values, axis=1)
    )


def load_tf_dictionary(species: str, base_grn: str) -> dict:
    """Return a shallow mapping copy so each Oracle owns its TF dictionary."""
    normalized_mode = "promoter" if base_grn in {"auto", "promoter"} else base_grn
    return _load_tf_dictionary_cached(species, normalized_mode).copy()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run CellOracle GRN inference.")
    parser.add_argument("--inFile", required=True, help="genes-by-cells expression CSV")
    parser.add_argument("--outFile", required=True, help="output ranked edge list (tsv)")
    parser.add_argument("--species", default="human")
    parser.add_argument("--baseGrn", default="auto",
                        help="'auto'/'promoter' for the species promoter GRN, or 'mouse_scATAC_atlas'")
    parser.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization for get_links")
    parser.add_argument("--pValueCutoff", type=float, default=0.05)
    parser.add_argument("--topK", type=int, default=0, help="max regulators per target (0 = keep all)")
    parser.add_argument("--maxGenes", type=int, default=0, help="cap to top-variance genes (0 = no cap)")
    parser.add_argument("--maxCells", type=int, default=0, help="subsample cells (0 = no cap)")
    parser.add_argument("--nJobs", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args(argv)


def build_anndata(expression: pd.DataFrame, max_genes: int, max_cells: int) -> ad.AnnData:
    # Optional gene cap by variance (CellOracle is memory-heavy on large gene sets).
    if max_genes and expression.shape[0] > max_genes:
        top_genes = expression.var(axis=1).sort_values(ascending=False).index[:max_genes]
        expression = expression.loc[top_genes]

    # AnnData is cells-by-genes; the input is genes-by-cells.
    adata = ad.AnnData(
        X=expression.T.to_numpy(dtype=np.float32),
        obs=pd.DataFrame(index=expression.columns.astype(str)),
        var=pd.DataFrame(index=expression.index.astype(str)),
    )

    if max_cells and adata.n_obs > max_cells:
        # Retain CellOracle's existing deterministic maxCells selection exactly.
        sc.pp.subsample(adata, n_obs=max_cells, random_state=0)
    return adata


def preprocess_for_celloracle(adata: ad.AnnData):
    """Attach only metadata required by Oracle's raw-count import.

    Oracle performs its own log normalization, PCA, and KNN imputation.  The
    former Scanpy normalization/PCA/neighbors/UMAP pass was therefore repeated
    work: its only consumed result was the two-dimensional visualization
    embedding.  A deterministic circular placeholder preserves that API
    contract without entering the GRN inference calculations.
    """
    n_comps = int(min(50, adata.n_obs - 1, adata.n_vars - 1))
    n_comps = max(2, n_comps)

    # One GRN unit for this scope (per-cluster scoping is done upstream).
    adata.obs[GRN_UNIT_COLUMN] = "unit"
    adata.obs[GRN_UNIT_COLUMN] = adata.obs[GRN_UNIT_COLUMN].astype("category")
    angles = np.linspace(0.0, 2.0 * np.pi, adata.n_obs, endpoint=False)
    adata.obsm["X_grnscope"] = np.column_stack(
        (np.cos(angles), np.sin(angles))
    ).astype(np.float32)
    return adata, n_comps, "X_grnscope"


def infer_links(
    adata: ad.AnnData,
    tf_dictionary: dict,
    n_comps: int,
    embedding_name: str,
    alpha: float,
    n_jobs: int,
):
    oracle = co.Oracle()
    oracle.import_anndata_as_raw_count(
        adata=adata,
        cluster_column_name=GRN_UNIT_COLUMN,
        embedding_name=embedding_name,
    )
    oracle.import_TF_data(TFdict=tf_dictionary)
    oracle.perform_PCA()

    n_cell = oracle.adata.shape[0]
    k = max(1, int(0.025 * n_cell))
    oracle.knn_imputation(
        n_pca_dims=int(min(50, n_comps)),
        k=k,
        balanced=True,
        b_sight=int(min(k * 8, n_cell - 1)),
        b_maxl=int(min(k * 4, n_cell - 1)),
        n_jobs=n_jobs,
    )

    links = oracle.get_links(
        cluster_name_for_GRN_unit=GRN_UNIT_COLUMN,
        alpha=alpha,
        bagging_number=20,
        verbose_level=0,
        n_jobs=n_jobs,
    )
    return links


def extract_edges(links, p_value_cutoff: float, top_k: int) -> pd.DataFrame:
    # Single GRN unit -> take the only per-cluster table.
    edges = list(links.links_dict.values())[0].copy()
    # Keep significant edges; CellOracle columns: source, target, coef_mean, coef_abs, p, -logp.
    edges = edges[edges["p"] <= p_value_cutoff]
    edges = edges[edges["coef_abs"] > 0]

    if top_k and not edges.empty:
        edges = (
            edges.sort_values("coef_abs", ascending=False)
            .groupby("target", group_keys=False)
            .head(top_k)
        )

    edges = edges.sort_values("coef_abs", ascending=False)
    # Signed regulatory coefficient as edge weight (CellOracle is directed + signed).
    return pd.DataFrame({
        "Gene1": edges["source"].astype(str),
        "Gene2": edges["target"].astype(str),
        "EdgeWeight": edges["coef_mean"].astype(float),
    })


def execute_inference(args):
    expression = pd.read_csv(args.inFile, index_col=0)
    if expression.empty:
        raise ValueError("Expression matrix is empty.")

    adata = build_anndata(expression, args.maxGenes, args.maxCells)
    adata, n_comps, embedding_name = preprocess_for_celloracle(adata)

    cache_hits_before = _load_tf_dictionary_cached.cache_info().hits
    tf_dictionary = load_tf_dictionary(args.species, args.baseGrn)
    prior_cache_hit = (
        _load_tf_dictionary_cached.cache_info().hits > cache_hits_before
    )
    links = infer_links(
        adata,
        tf_dictionary,
        n_comps,
        embedding_name,
        args.alpha,
        args.nJobs,
    )
    ranked = extract_edges(links, args.pValueCutoff, args.topK)

    ranked.to_csv(args.outFile, sep="\t", header=True, index=False)
    print(f"CellOracle wrote {len(ranked)} edges to {args.outFile}")
    return {
        "edge_count": int(len(ranked)),
        "prior_cache_hit": prior_cache_hit,
        "preprocessing_mode": "oracle_native_with_metadata_embedding",
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _format_elapsed(seconds: float) -> str:
    minutes, remaining_seconds = divmod(max(0.0, seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}:{minutes:02d}:{remaining_seconds:05.2f}"


def _write_time_metrics(
    path: Path,
    *,
    started_at: float,
    usage_before,
    usage_after,
) -> None:
    elapsed = time.perf_counter() - started_at
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"User time (seconds): {max(0.0, usage_after.ru_utime - usage_before.ru_utime):.6f}",
                f"System time (seconds): {max(0.0, usage_after.ru_stime - usage_before.ru_stime):.6f}",
                f"Elapsed (wall clock) time (h:mm:ss or m:ss): {_format_elapsed(elapsed)}",
                f"Maximum resident set size (kbytes): {int(usage_after.ru_maxrss)}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_worker() -> None:
    """Accept JSON-line requests while keeping CellOracle and priors resident."""
    request_index = 0
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        request_index += 1
        request = {}
        response_path = None
        log_path = None
        started_at = time.perf_counter()
        usage_before = resource.getrusage(resource.RUSAGE_SELF)
        try:
            request = json.loads(raw_line)
            response_path = Path(request["responseFile"])
            log_path = Path(request["logFile"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            args = parse_args(request["argv"])
            with log_path.open("w", encoding="utf-8") as log_file:
                with redirect_stdout(log_file), redirect_stderr(log_file):
                    telemetry = execute_inference(args)
            response = {
                "status": "Completed",
                "request_id": request.get("request_id"),
                "worker_pid": os.getpid(),
                "request_index": request_index,
                "elapsed_seconds": round(time.perf_counter() - started_at, 6),
                **telemetry,
            }
        except Exception as exc:
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as log_file:
                    traceback.print_exc(file=log_file)
            else:
                traceback.print_exc()
            response = {
                "status": "Failed",
                "request_id": request.get("request_id"),
                "worker_pid": os.getpid(),
                "request_index": request_index,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "elapsed_seconds": round(time.perf_counter() - started_at, 6),
            }

        usage_after = resource.getrusage(resource.RUSAGE_SELF)
        time_file_value = request.get("timeFile")
        if time_file_value:
            _write_time_metrics(
                Path(time_file_value),
                started_at=started_at,
                usage_before=usage_before,
                usage_after=usage_after,
            )
        if response_path is not None:
            _write_json_atomic(response_path, response)


def main(argv):
    if "--worker" in argv:
        run_worker()
        return
    args = parse_args(argv)
    execute_inference(args)


if __name__ == "__main__":
    main(sys.argv[1:])
