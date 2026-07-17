"""Prepare and run a CellOracle perturbation from a completed GRNScope network.

The completed CellOracle edge list supplies the TF -> target topology.  This
script refits CellOracle's simulation coefficients from the project's expression
matrix, caches the simulation-ready Oracle object, and runs one perturbation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import anndata as ad
import celloracle as co
import numpy as np
import pandas as pd
import scanpy as sc
from celloracle.applications import Gradient_calculator, Oracle_development_module

from perturbation_distributions import summarize_expression_distribution


GRN_UNIT_COLUMN = "grn_unit"
CLUSTER_COLUMN = "cluster"
MIN_CELLS_PER_CLUSTER_SAMPLE = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CellOracle perturbation.")
    parser.add_argument("--expression", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--clusters")
    parser.add_argument("--pseudotime")
    parser.add_argument("--cluster-edges-manifest")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gene", required=True)
    parser.add_argument("--value", type=float, default=0.0)
    parser.add_argument("--n-propagation", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--max-cells", type=int, default=2000)
    parser.add_argument("--clip-delta-x", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args()


def read_table(path: Path, *, index_col: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", index_col=index_col)


def read_cluster_labels(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(65536)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(handle, dialect=dialect))

    if not rows:
        return {}

    first = [str(value).strip().lower() for value in rows[0][:2]]
    has_header = len(first) >= 2 and first[0] in {
        "cell", "cell_id", "cellid", "cell id", "cells"
    }
    data_rows = rows[1:] if has_header else rows
    return {
        str(row[0]).strip(): str(row[1]).strip()
        for row in data_rows
        if len(row) >= 2 and str(row[0]).strip() and str(row[1]).strip()
    }


def find_column(columns: list[str], candidates: set[str], fallback: int) -> str:
    for column in columns:
        if column.strip().lower() in candidates:
            return column
    if len(columns) <= fallback:
        raise ValueError(f"Could not identify edge-list columns: {columns}")
    return columns[fallback]


def build_tf_dictionary(edges_path: Path, expression_genes: set[str]) -> dict[str, list[str]]:
    edges = read_table(edges_path)
    columns = [str(column) for column in edges.columns]
    source_column = find_column(columns, {"gene1", "tf", "source", "regulator"}, 0)
    target_column = find_column(columns, {"gene2", "target", "targetgene"}, 1)

    tf_dict: dict[str, set[str]] = {}
    for source_raw, target_raw in edges[[source_column, target_column]].itertuples(index=False):
        source = str(source_raw).strip()
        target = str(target_raw).strip()
        if not source or not target or source not in expression_genes or target not in expression_genes:
            continue
        tf_dict.setdefault(target, set()).add(source)

    if not tf_dict:
        raise ValueError("No CellOracle edges overlap the expression matrix.")
    return {target: sorted(regulators) for target, regulators in tf_dict.items()}


def read_cluster_edge_paths(path: Path | None) -> dict[str, Path]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Cluster edge manifest must be a label-to-path object.")
    paths: dict[str, Path] = {}
    for label, raw_path in payload.items():
        normalized_label = str(label).strip()
        candidate = Path(str(raw_path))
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        if normalized_label and candidate.is_file():
            paths[normalized_label] = candidate
    return paths


def build_anndata(
    expression: pd.DataFrame,
    cluster_labels: dict[str, str],
    max_cells: int,
) -> ad.AnnData:
    expression.columns = expression.columns.astype(str)
    expression.index = expression.index.astype(str)
    adata = ad.AnnData(
        X=expression.T.to_numpy(dtype=np.float32),
        obs=pd.DataFrame(index=expression.columns),
        var=pd.DataFrame(index=expression.index),
    )
    adata.uns["grnscope_input_cell_count"] = int(adata.n_obs)
    clusters = np.asarray(
        [cluster_labels.get(str(cell_id), "Unlabelled") for cell_id in adata.obs_names],
        dtype=str,
    )
    matched_labels = sum(str(cell_id) in cluster_labels for cell_id in adata.obs_names)
    adata.obs[CLUSTER_COLUMN] = pd.Categorical(clusters)
    adata.obs[GRN_UNIT_COLUMN] = pd.Categorical(["Global"] * adata.n_obs)
    adata.uns["grnscope_cluster_labels_matched"] = int(matched_labels)

    if max_cells > 0 and adata.n_obs > max_cells:
        rng = np.random.default_rng(0)
        selected_seed: list[int] = []
        if matched_labels:
            cluster_indices = {
                cluster: np.flatnonzero(clusters == cluster)
                for cluster in sorted(np.unique(clusters))
            }
            minimum_total = sum(
                min(MIN_CELLS_PER_CLUSTER_SAMPLE, len(indices))
                for indices in cluster_indices.values()
            )
            if minimum_total <= max_cells:
                for indices in cluster_indices.values():
                    take = min(MIN_CELLS_PER_CLUSTER_SAMPLE, len(indices))
                    selected_seed.extend(rng.choice(indices, size=take, replace=False).tolist())
            elif len(cluster_indices) <= max_cells:
                for indices in cluster_indices.values():
                    selected_seed.append(int(rng.choice(indices)))
        selected_set = set(selected_seed)
        remaining_pool = np.asarray(
            [index for index in range(adata.n_obs) if index not in selected_set],
            dtype=int,
        )
        remaining_count = max_cells - len(selected_set)
        selected_extra = (
            rng.choice(remaining_pool, size=remaining_count, replace=False).tolist()
            if remaining_count > 0
            else []
        )
        selected = np.sort(np.asarray([*selected_set, *selected_extra], dtype=int))
        adata = adata[selected].copy()
    return adata


def prepare_oracle(
    expression_path: Path,
    edges_path: Path,
    clusters_path: Path | None,
    cluster_edges_manifest_path: Path | None,
    model_path: Path,
    alpha: float,
    max_cells: int,
    n_jobs: int,
):
    expression = read_table(expression_path, index_col=0)
    if expression.empty:
        raise ValueError("Expression matrix is empty.")
    if expression.shape[0] < 2 or expression.shape[1] < 3:
        raise ValueError("CellOracle perturbation requires at least two genes and three cells.")

    tf_dict = build_tf_dictionary(edges_path, set(expression.index.astype(str)))
    labels = read_cluster_labels(clusters_path)
    adata = build_anndata(expression, labels, max_cells)

    adata.layers["raw_count"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.scale(adata)

    n_comps = max(2, int(min(50, adata.n_obs - 1, adata.n_vars - 1)))
    sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")
    sc.pp.neighbors(
        adata,
        n_neighbors=int(min(15, max(2, adata.n_obs - 1))),
        n_pcs=n_comps,
    )
    sc.tl.umap(adata, random_state=0)
    adata.X = adata.layers["raw_count"].copy()

    use_cluster_grn = bool(int(adata.uns.get("grnscope_cluster_labels_matched", 0)))
    grn_unit = "cluster" if use_cluster_grn else "whole"
    cluster_column = CLUSTER_COLUMN if use_cluster_grn else GRN_UNIT_COLUMN

    oracle = co.Oracle()
    oracle.import_anndata_as_raw_count(
        adata=adata,
        cluster_column_name=cluster_column,
        embedding_name="X_umap",
    )
    oracle.import_TF_data(TFdict=tf_dict)
    oracle.perform_PCA()

    n_cells = oracle.adata.shape[0]
    k = max(1, int(0.025 * n_cells))
    oracle.knn_imputation(
        n_pca_dims=int(min(50, n_comps)),
        k=k,
        balanced=True,
        b_sight=int(min(k * 8, n_cells - 1)),
        b_maxl=int(min(k * 4, n_cells - 1)),
        n_jobs=n_jobs,
    )
    cluster_edge_paths = read_cluster_edge_paths(cluster_edges_manifest_path)
    cluster_topology_labels: list[str] = []
    fallback_topology_labels: list[str] = []
    if use_cluster_grn:
        oracle.cluster_specific_TFdict = {}
        for cluster in oracle.adata.obs[CLUSTER_COLUMN].astype(str).unique():
            cluster_path = cluster_edge_paths.get(cluster)
            if cluster_path is not None:
                oracle.cluster_specific_TFdict[cluster] = build_tf_dictionary(
                    cluster_path,
                    set(expression.index.astype(str)),
                )
                cluster_topology_labels.append(cluster)
            else:
                oracle.cluster_specific_TFdict[cluster] = tf_dict
                fallback_topology_labels.append(cluster)

    oracle.fit_GRN_for_simulation(
        GRN_unit=grn_unit,
        alpha=alpha,
        use_cluster_specific_TFdict=use_cluster_grn,
        verbose_level=0,
    )
    oracle.adata.uns["grnscope_grn_unit"] = grn_unit
    oracle.adata.uns["grnscope_cluster_count"] = (
        int(oracle.adata.obs[CLUSTER_COLUMN].nunique()) if use_cluster_grn else 0
    )
    oracle.adata.uns["grnscope_cluster_topology_labels"] = sorted(cluster_topology_labels)
    oracle.adata.uns["grnscope_fallback_topology_labels"] = sorted(fallback_topology_labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    oracle.to_hdf5(str(model_path))
    return oracle


def finite_float(value: object) -> float:
    number = float(value)
    return number if np.isfinite(number) else 0.0


def write_expression_limits(oracle, model_path: Path) -> None:
    imputed = oracle.adata.to_df(layer="imputed_count")
    limits = {}
    for gene in imputed.columns:
        minimum = finite_float(imputed[gene].min())
        maximum = finite_float(imputed[gene].max())
        limits[str(gene)] = {
            "minimum": minimum,
            "maximum": maximum,
            "safe_upper_limit": maximum + (maximum - minimum),
        }
    path = model_path.parent / "expression_limits.json"
    path.write_text(
        json.dumps({"source": "celloracle_imputed_count", "genes": limits}, indent=2),
        encoding="utf-8",
    )


def sampled_indices(length: int, limit: int) -> np.ndarray:
    if length <= limit:
        return np.arange(length)
    return np.linspace(0, length - 1, limit, dtype=int)


PSEUDOTIME_KEY = "GRNScopePseudotime"
PSEUDOTIME_MISSING_TOKENS = {"", "na", "nan", "null", "none"}
CELL_ID_HEADERS = {"cell", "cell_id", "cellid", "cell id", "cells"}


def optional_finite_float(value: object) -> float | None:
    text = str(value).strip()
    if text.lower() in PSEUDOTIME_MISSING_TOKENS:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def unique_trajectory_names(raw_names: list[str]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for index, raw_name in enumerate(raw_names, start=1):
        base = str(raw_name).strip() or f"PseudoTime{index}"
        name = base
        suffix = 2
        while name in seen:
            name = f"{base}_{suffix}"
            suffix += 1
        names.append(name)
        seen.add(name)
    return names


def read_pseudotime_trajectories(
    path: Path,
    expression_cell_ids: list[str],
) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(65536)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        rows = [
            [str(value).strip() for value in row]
            for row in csv.reader(handle, dialect=dialect)
            if any(str(value).strip() for value in row)
        ]

    if not rows:
        raise ValueError("Pseudotime file is empty.")

    if len(rows[0]) == 1:
        first_is_value = optional_finite_float(rows[0][0]) is not None
        name = "Pseudotime" if first_is_value else (rows[0][0] or "Pseudotime")
        data_rows = rows if first_is_value else rows[1:]
        if len(data_rows) != len(expression_cell_ids):
            raise ValueError("Pseudotime row count does not match the expression cells.")
        values: dict[str, float] = {}
        for cell_id, row in zip(expression_cell_ids, data_rows):
            value = optional_finite_float(row[0] if row else "")
            if value is None:
                raise ValueError("Single-column pseudotime contains a missing value.")
            values[cell_id] = value
        return {name: values}

    headers = rows[0]
    data_rows = rows[1:]
    first_data_width = len(data_rows[0]) if data_rows else len(headers)
    explicit_cell_header = headers[0].strip().lower() in CELL_ID_HEADERS or headers[0] == ""
    shifted_headers = (
        not explicit_cell_header
        and (
            first_data_width == len(headers) + 1
            or (headers[-1] == "" and first_data_width == len(headers))
        )
    )
    if shifted_headers:
        raw_trajectory_names = (
            headers[:-1]
            if headers[-1] == "" and first_data_width == len(headers)
            else headers
        )
        expected_width = len(headers) if headers[-1] == "" else len(headers) + 1
    else:
        raw_trajectory_names = headers[1:]
        expected_width = len(headers)

    trajectory_names = unique_trajectory_names(raw_trajectory_names)
    trajectories = {name: {} for name in trajectory_names}
    for row in data_rows:
        padded = [*row, *([""] * max(0, expected_width - len(row)))]
        cell_id = padded[0].strip()
        if not cell_id:
            continue
        for name, raw_value in zip(trajectory_names, padded[1 : len(trajectory_names) + 1]):
            value = optional_finite_float(raw_value)
            if value is not None:
                trajectories[name][cell_id] = value
    return trajectories


def load_best_pseudotime_trajectory(
    oracle,
    pseudotime_path: Path,
    expression_path: Path,
) -> tuple[str, np.ndarray]:
    expression_headers = pd.read_csv(
        expression_path,
        sep=None,
        engine="python",
        nrows=0,
        index_col=0,
    )
    expression_cell_ids = [str(cell_id) for cell_id in expression_headers.columns]
    trajectories = read_pseudotime_trajectories(pseudotime_path, expression_cell_ids)
    oracle_cell_ids = [str(cell_id) for cell_id in oracle.adata.obs_names]
    ranked = sorted(
        trajectories.items(),
        key=lambda item: -sum(cell_id in item[1] for cell_id in oracle_cell_ids),
    )
    if not ranked:
        raise ValueError("Pseudotime file contains no trajectories.")
    trajectory_name, values_by_cell = ranked[0]
    pseudotime = np.asarray(
        [values_by_cell.get(cell_id, np.nan) for cell_id in oracle_cell_ids],
        dtype=float,
    )
    usable_indices = np.flatnonzero(np.isfinite(pseudotime))
    if len(usable_indices) < 20:
        raise ValueError("Pseudotime trajectory has fewer than 20 analyzed cells.")
    oracle.adata.obs[PSEUDOTIME_KEY] = pseudotime
    return trajectory_name, usable_indices


def unavailable_ps_metrics(reason: str) -> dict:
    return {
        "perturbation_score": None,
        "perturbation_score_p_value": None,
        "perturbation_score_direction": None,
        "perturbation_score_grid_point_count": 0,
        "perturbation_score_unavailable_reason": reason,
    }


def calculate_ps_metrics(oracle, cell_indices: np.ndarray, n_jobs: int) -> dict:
    if len(cell_indices) < 20:
        return unavailable_ps_metrics("Too few cells have pseudotime in this scope.")

    n_neighbors = int(min(200, max(2, len(cell_indices) - 2)))
    gradient = Gradient_calculator(
        oracle_object=oracle,
        pseudotime_key=PSEUDOTIME_KEY,
        cell_idx_use=cell_indices,
    )
    gradient.calculate_p_mass(
        smooth=0.8,
        n_grid=40,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
    )
    gradient.calculate_mass_filter(min_mass=0.01, plot=False)
    gradient.transfer_data_into_grid(args={"method": "polynomial", "n_poly": 3})
    gradient.calculate_gradient()

    development = Oracle_development_module()
    development.load_differentiation_reference_data(gradient_object=gradient)
    development.load_perturb_simulation_data(
        oracle_object=oracle,
        cell_idx_use=cell_indices,
        n_neighbors=n_neighbors,
    )
    development.calculate_inner_product()
    development.calculate_digitized_ip(n_bins=10)

    scores = development.inner_product_df["score"].to_numpy(dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) < 2:
        return unavailable_ps_metrics("Too few valid grid points for Perturbation Score.")

    perturbation_score = float(scores.mean())
    if perturbation_score > 0:
        direction = "promotes"
        p_value = development.get_positive_PS_p_value()
    elif perturbation_score < 0:
        direction = "blocks"
        p_value = development.get_negative_PS_p_value()
    else:
        direction = "neutral"
        p_value = None

    p_value = optional_finite_float(p_value) if p_value is not None else None
    return {
        "perturbation_score": perturbation_score,
        "perturbation_score_p_value": p_value,
        "perturbation_score_direction": direction,
        "perturbation_score_grid_point_count": int(len(scores)),
        "perturbation_score_unavailable_reason": (
            None if p_value is not None else "The paired Wilcoxon test could not be calculated."
        ),
    }


def attach_perturbation_score_metrics(
    result: dict,
    oracle,
    args: argparse.Namespace,
) -> None:
    if not args.pseudotime:
        result.update(unavailable_ps_metrics("Pseudotime was not uploaded for this project."))
        for summary in result.get("cluster_summary", []):
            summary.update(unavailable_ps_metrics("Pseudotime was not uploaded for this project."))
        return

    try:
        trajectory_name, trajectory_indices = load_best_pseudotime_trajectory(
            oracle,
            Path(args.pseudotime),
            Path(args.expression),
        )
    except Exception as exc:
        reason = f"Pseudotime could not be used: {exc}"
        result.update(unavailable_ps_metrics(reason))
        for summary in result.get("cluster_summary", []):
            summary.update(unavailable_ps_metrics(reason))
        return

    result["pseudotime_trajectory"] = trajectory_name
    result["pseudotime_cell_count"] = int(len(trajectory_indices))
    try:
        result.update(calculate_ps_metrics(oracle, trajectory_indices, args.n_jobs))
    except Exception as exc:
        result.update(unavailable_ps_metrics(f"Perturbation Score calculation failed: {exc}"))

    clusters = oracle.adata.obs[CLUSTER_COLUMN].astype(str).to_numpy()
    trajectory_mask = np.zeros(len(clusters), dtype=bool)
    trajectory_mask[trajectory_indices] = True
    for summary in result.get("cluster_summary", []):
        cluster_indices = np.flatnonzero(
            trajectory_mask & (clusters == str(summary.get("cluster", "")))
        )
        try:
            summary.update(calculate_ps_metrics(oracle, cluster_indices, args.n_jobs))
        except Exception as exc:
            summary.update(unavailable_ps_metrics(f"Perturbation Score calculation failed: {exc}"))


def build_result(oracle, args: argparse.Namespace, model_reused: bool) -> dict:
    delta = oracle.adata.to_df(layer="delta_X")
    original = oracle.adata.to_df(layer="imputed_count")
    simulated = oracle.adata.to_df(layer="simulated_count")
    ood = oracle.evaluate_simulated_gene_distribution_range()

    affected = pd.DataFrame(
        {
            "gene": delta.columns,
            "mean_change": delta.mean(axis=0).values,
            "mean_absolute_change": delta.abs().mean(axis=0).values,
            "original_mean": original.mean(axis=0).values,
            "simulated_mean": simulated.mean(axis=0).values,
        }
    ).sort_values("mean_absolute_change", ascending=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    affected.to_csv(output_dir / "affected_genes.csv", index=False)

    embedding = np.asarray(oracle.embedding)
    shifts = np.asarray(oracle.delta_embedding)
    random_shifts = np.asarray(oracle.delta_embedding_random)
    shift_distances = np.linalg.norm(shifts - random_shifts, axis=1)
    clusters = oracle.adata.obs[CLUSTER_COLUMN].astype(str).to_numpy()
    cells = oracle.adata.obs_names.astype(str).to_numpy()

    cell_table = pd.DataFrame(
        {
            "cell_id": cells,
            "cluster": clusters,
            "embedding_x": embedding[:, 0],
            "embedding_y": embedding[:, 1],
            "shift_x": shifts[:, 0],
            "shift_y": shifts[:, 1],
            "random_shift_x": random_shifts[:, 0],
            "random_shift_y": random_shifts[:, 1],
            "shift_distance": shift_distances,
        }
    )
    cell_table.to_csv(output_dir / "cell_shifts.csv", index=False)

    cluster_summary = (
        pd.DataFrame(delta.assign(cluster=clusters))
        .groupby("cluster", observed=True)
        .mean()
    )
    cluster_cell_counts = pd.Series(clusters).value_counts().to_dict()
    cluster_shift_magnitudes = (
        pd.DataFrame(
            {
                "cluster": clusters,
                "shift_magnitude": np.linalg.norm(shifts, axis=1),
                "random_shift_magnitude": np.linalg.norm(random_shifts, axis=1),
            }
        )
        .groupby("cluster", observed=True)
        .mean()
    )
    cluster_rows: list[dict] = []
    cluster_summaries: list[dict] = []
    for cluster, row in cluster_summary.iterrows():
        cluster_label = str(cluster)
        shift_magnitude = finite_float(cluster_shift_magnitudes.loc[cluster, "shift_magnitude"])
        random_shift_magnitude = finite_float(
            cluster_shift_magnitudes.loc[cluster, "random_shift_magnitude"]
        )
        cluster_summaries.append(
            {
                "cluster": cluster_label,
                "cell_count": int(cluster_cell_counts.get(cluster_label, 0)),
                "mean_shift_magnitude": shift_magnitude,
                "mean_random_shift_magnitude": random_shift_magnitude,
                "shift_ratio": (
                    shift_magnitude / random_shift_magnitude
                    if random_shift_magnitude > 0
                    else None
                ),
            }
        )
        for gene, value in row.abs().nlargest(10).items():
            cluster_rows.append(
                {
                    "cluster": cluster_label,
                    "gene": str(gene),
                    "mean_change": finite_float(cluster_summary.loc[cluster, gene]),
                }
            )
    cluster_effects_by_cluster = {}
    for effect in cluster_rows:
        cluster_effects_by_cluster.setdefault(effect["cluster"], []).append(
            {
                "gene": effect["gene"],
                "mean_change": effect["mean_change"],
            }
        )
    for summary in cluster_summaries:
        summary["top_genes"] = cluster_effects_by_cluster.get(summary["cluster"], [])[:5]
    pd.DataFrame(cluster_rows).to_csv(output_dir / "cluster_effects.csv", index=False)

    gene_expression_distributions: list[dict] = []
    for gene in affected.head(25)["gene"].astype(str):
        gene_expression_distributions.append(
            summarize_expression_distribution(
                gene=gene,
                baseline_values=original[gene].to_numpy(),
                simulated_values=simulated[gene].to_numpy(),
                scope_type="global",
            )
        )
    for cluster_label, effects in cluster_effects_by_cluster.items():
        cluster_mask = clusters == cluster_label
        for effect in effects[:10]:
            gene = str(effect["gene"])
            gene_expression_distributions.append(
                summarize_expression_distribution(
                    gene=gene,
                    baseline_values=original.loc[cluster_mask, gene].to_numpy(),
                    simulated_values=simulated.loc[cluster_mask, gene].to_numpy(),
                    scope_type="cluster",
                    scope_label=cluster_label,
                )
            )

    point_ids = sampled_indices(len(cells), 3000)
    ood_warnings = ood[ood["Max_exceeding_ratio"] > 0.5]
    ood_warning_count = int(len(ood_warnings))
    ood_export = ood_warnings.reset_index()
    ood_export.columns = ["gene", "max_exceeding_ratio", "ood_cell_ratio"]
    ood_export.to_csv(output_dir / "ood_diagnostics.csv", index=False)
    mass_filter = np.asarray(oracle.mass_filter, dtype=bool)
    grid = np.asarray(oracle.flow_grid)
    flow = np.asarray(oracle.flow)
    randomized_flow = np.asarray(oracle.flow_rndm)
    density = np.asarray(oracle.total_p_mass)
    selected_grid = np.flatnonzero(~mass_filter)
    grn_unit = str(oracle.adata.uns.get("grnscope_grn_unit", "whole"))
    cluster_topology_labels = [
        str(label)
        for label in oracle.adata.uns.get("grnscope_cluster_topology_labels", [])
    ]
    fallback_topology_labels = [
        str(label)
        for label in oracle.adata.uns.get("grnscope_fallback_topology_labels", [])
    ]

    return {
        "gene": args.gene,
        "perturbation_value": args.value,
        "n_propagation": args.n_propagation,
        "clip_delta_x": bool(args.clip_delta_x),
        "celloracle_version": str(co.__version__),
        "model_reused": model_reused,
        "model_scope": "cluster_specific" if grn_unit == "cluster" else "global",
        "grn_unit": grn_unit,
        "cluster_count": int(oracle.adata.uns.get("grnscope_cluster_count", 0)),
        "cluster_specific_topology_count": len(cluster_topology_labels),
        "cluster_specific_topology_labels": cluster_topology_labels,
        "global_topology_fallback_labels": fallback_topology_labels,
        "input_cells": int(oracle.adata.uns.get("grnscope_input_cell_count", len(cells))),
        "cells_analyzed": int(len(cells)),
        "genes_analyzed": int(delta.shape[1]),
        "ood_warning_gene_count": ood_warning_count,
        "max_ood_exceeding_ratio": finite_float(ood["Max_exceeding_ratio"].max()),
        "ood_genes": [
            {
                "gene": str(gene),
                "max_exceeding_ratio": finite_float(row["Max_exceeding_ratio"]),
                "ood_cell_ratio": finite_float(row["OOD_cell_ratio"]),
            }
            for gene, row in ood_warnings.iterrows()
        ],
        "mean_shift_magnitude": finite_float(np.linalg.norm(shifts, axis=1).mean()),
        "mean_random_shift_magnitude": finite_float(np.linalg.norm(random_shifts, axis=1).mean()),
        "top_affected_genes": [
            {
                "gene": str(row.gene),
                "mean_change": finite_float(row.mean_change),
                "mean_absolute_change": finite_float(row.mean_absolute_change),
                "original_mean": finite_float(row.original_mean),
                "simulated_mean": finite_float(row.simulated_mean),
            }
            for row in affected.head(25).itertuples(index=False)
        ],
        "cluster_effects": cluster_rows,
        "cluster_summary": cluster_summaries,
        "gene_expression_distributions": gene_expression_distributions,
        "embedding_points": [
            {
                "x": finite_float(embedding[index, 0]),
                "y": finite_float(embedding[index, 1]),
                "cluster": str(clusters[index]),
                "shift_x": finite_float(shifts[index, 0]),
                "shift_y": finite_float(shifts[index, 1]),
                "random_shift_x": finite_float(random_shifts[index, 0]),
                "random_shift_y": finite_float(random_shifts[index, 1]),
            }
            for index in point_ids
        ],
        "grid_vectors": [
            {
                "x": finite_float(grid[index, 0]),
                "y": finite_float(grid[index, 1]),
                "dx": finite_float(flow[index, 0]),
                "dy": finite_float(flow[index, 1]),
                "random_dx": finite_float(randomized_flow[index, 0]),
                "random_dy": finite_float(randomized_flow[index, 1]),
                "density": finite_float(density[index]),
            }
            for index in selected_grid
        ],
        "grid_settings": {
            "source": "celloracle",
            "smooth": 0.8,
            "steps": [40, 40],
            "min_mass": 0.01,
            "n_neighbors": int(args.grid_n_neighbors),
        },
    }


def main() -> None:
    args = parse_args()
    if args.value < 0:
        raise ValueError("Perturbation value cannot be negative.")
    if not 1 <= args.n_propagation <= 5:
        raise ValueError("n-propagation must be between 1 and 5.")

    model_path = Path(args.model)
    model_reused = model_path.exists()
    if model_reused:
        oracle = co.load_hdf5(str(model_path))
    else:
        oracle = prepare_oracle(
            Path(args.expression),
            Path(args.edges),
            Path(args.clusters) if args.clusters else None,
            Path(args.cluster_edges_manifest) if args.cluster_edges_manifest else None,
            model_path,
            args.alpha,
            args.max_cells,
            args.n_jobs,
        )

    if args.gene not in oracle.adata.var_names:
        raise ValueError(f"Gene {args.gene} is not present in the perturbation model.")
    if args.gene not in oracle.active_regulatory_genes:
        raise ValueError(f"Gene {args.gene} is not an active CellOracle regulator.")

    write_expression_limits(oracle, model_path)

    grn_unit = str(oracle.adata.uns.get("grnscope_grn_unit", getattr(oracle, "GRN_unit", "whole")))
    oracle.simulate_shift(
        perturb_condition={args.gene: args.value},
        GRN_unit=grn_unit,
        n_propagation=args.n_propagation,
        clip_delta_X=args.clip_delta_x,
    )
    # CellOracle requests n_neighbors + 1 internally, so this must stay below
    # the fitted cell count.
    neighbor_count = int(min(200, max(1, oracle.adata.n_obs - 2)))
    oracle.estimate_transition_prob(
        n_neighbors=neighbor_count,
        knn_random=True,
        sampled_fraction=1,
        calculate_randomized=True,
        n_jobs=args.n_jobs,
        random_seed=0,
    )
    oracle.calculate_embedding_shift(sigma_corr=0.05)
    oracle.calculate_grid_arrows(
        smooth=0.8,
        steps=(40, 40),
        n_neighbors=neighbor_count,
        n_jobs=args.n_jobs,
    )
    oracle.calculate_mass_filter(min_mass=0.01, plot=False)
    args.grid_n_neighbors = neighbor_count

    result = build_result(oracle, args, model_reused)
    attach_perturbation_score_metrics(result, oracle, args)
    output_path = Path(args.output_dir) / "result.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"CellOracle perturbation result written to {output_path}")


if __name__ == "__main__":
    main()
