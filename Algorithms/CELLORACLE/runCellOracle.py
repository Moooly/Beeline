"""In-container CellOracle GRN inference for BEELINE.

Reads a genes-by-cells expression matrix, runs CellOracle's prior-informed GRN
inference on the given cells as a single GRN unit (per-cluster scoping is handled
upstream by GRNScope, which passes one cluster's cells at a time), and writes a
ranked directed/signed edge list.

Pipeline (CellOracle standard):
  expression -> AnnData -> scanpy preprocessing (normalize/log/PCA/neighbors/UMAP)
  -> Oracle.import_anndata_as_raw_count + import_TF_data(base GRN)
  -> perform_PCA + knn_imputation -> get_links -> filter by p-value -> edge list

The base GRN is a species-specific promoter network (or the mouse scATAC atlas),
loaded from CellOracle's bundled data (baked into the image at build time).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import celloracle as co


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


def load_base_grn(species: str, base_grn: str) -> pd.DataFrame:
    """Load the base GRN prior for the requested species / mode."""
    if base_grn == "mouse_scATAC_atlas":
        if species != "mouse":
            raise ValueError("mouse_scATAC_atlas base GRN is only available for mouse.")
        return co.data.load_mouse_scATAC_atlas_base_GRN()

    loader = PROMOTER_BASE_GRN_LOADERS.get(species)
    if loader is None:
        raise ValueError(f"No built-in CellOracle base GRN for species '{species}'.")
    return loader()


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
        sc.pp.subsample(adata, n_obs=max_cells, random_state=0)
    return adata


def preprocess_for_celloracle(adata: ad.AnnData):
    # Keep raw counts; CellOracle re-imports them via import_anndata_as_raw_count.
    adata.layers["raw_count"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.scale(adata)

    n_comps = int(min(50, adata.n_obs - 1, adata.n_vars - 1))
    n_comps = max(2, n_comps)
    sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=int(min(15, max(2, adata.n_obs - 1))), n_pcs=n_comps)
    sc.tl.umap(adata)

    # One GRN unit for this scope (per-cluster scoping is done upstream).
    adata.obs[GRN_UNIT_COLUMN] = "unit"
    adata.obs[GRN_UNIT_COLUMN] = adata.obs[GRN_UNIT_COLUMN].astype("category")

    # Restore raw counts to X for CellOracle's import step.
    adata.X = adata.layers["raw_count"].copy()
    return adata, n_comps


def infer_links(adata: ad.AnnData, base_grn: pd.DataFrame, n_comps: int, alpha: float, n_jobs: int):
    oracle = co.Oracle()
    oracle.import_anndata_as_raw_count(
        adata=adata,
        cluster_column_name=GRN_UNIT_COLUMN,
        embedding_name="X_umap",
    )
    oracle.import_TF_data(TF_info_matrix=base_grn)
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


def main(argv):
    args = parse_args(argv)

    expression = pd.read_csv(args.inFile, index_col=0)
    if expression.empty:
        raise ValueError("Expression matrix is empty.")

    adata = build_anndata(expression, args.maxGenes, args.maxCells)
    adata, n_comps = preprocess_for_celloracle(adata)

    base_grn = load_base_grn(args.species, args.baseGrn)
    links = infer_links(adata, base_grn, n_comps, args.alpha, args.nJobs)
    ranked = extract_edges(links, args.pValueCutoff, args.topK)

    ranked.to_csv(args.outFile, sep="\t", header=True, index=False)
    print(f"CellOracle wrote {len(ranked)} edges to {args.outFile}")


if __name__ == "__main__":
    main(sys.argv[1:])
