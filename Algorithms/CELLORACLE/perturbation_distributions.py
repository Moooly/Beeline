"""Compact, model-scale expression summaries for CellOracle perturbation results."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


DEFAULT_BIN_COUNT = 12


def summarize_expression_distribution(
    *,
    gene: str,
    baseline_values: Sequence[float] | np.ndarray,
    simulated_values: Sequence[float] | np.ndarray,
    scope_type: str,
    scope_label: str | None = None,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> dict:
    """Summarize paired imputed and simulated expression values using shared bins."""
    baseline = np.asarray(baseline_values, dtype=float).reshape(-1)
    simulated = np.asarray(simulated_values, dtype=float).reshape(-1)
    if baseline.shape != simulated.shape:
        raise ValueError("Baseline and simulated expression arrays must have the same shape.")

    finite_mask = np.isfinite(baseline) & np.isfinite(simulated)
    baseline = baseline[finite_mask]
    simulated = simulated[finite_mask]
    if baseline.size == 0:
        raise ValueError("Expression distribution requires at least one finite cell pair.")

    delta = simulated - baseline
    combined_minimum = float(min(baseline.min(), simulated.min()))
    combined_maximum = float(max(baseline.max(), simulated.max()))
    scale = max(1.0, abs(combined_minimum), abs(combined_maximum))
    change_tolerance = scale * 1e-9

    if combined_minimum == combined_maximum:
        histogram = [
            {
                "start": combined_minimum,
                "end": combined_maximum,
                "baseline_count": int(baseline.size),
                "simulated_count": int(simulated.size),
            }
        ]
    else:
        edges = np.linspace(
            combined_minimum,
            combined_maximum,
            max(1, int(bin_count)) + 1,
        )
        baseline_counts, _ = np.histogram(baseline, bins=edges)
        simulated_counts, _ = np.histogram(simulated, bins=edges)
        histogram = [
            {
                "start": float(edges[index]),
                "end": float(edges[index + 1]),
                "baseline_count": int(baseline_counts[index]),
                "simulated_count": int(simulated_counts[index]),
            }
            for index in range(len(edges) - 1)
        ]

    return {
        "gene": str(gene),
        "scope_type": str(scope_type),
        "scope_label": str(scope_label) if scope_label is not None else None,
        "expression_layer": "celloracle_imputed_count",
        "cell_count": int(baseline.size),
        "baseline_mean": float(baseline.mean()),
        "simulated_mean": float(simulated.mean()),
        "baseline_median": float(np.median(baseline)),
        "simulated_median": float(np.median(simulated)),
        "mean_change": float(delta.mean()),
        "mean_absolute_change": float(np.abs(delta).mean()),
        "increased_cell_fraction": float(np.mean(delta > change_tolerance)),
        "decreased_cell_fraction": float(np.mean(delta < -change_tolerance)),
        "histogram": histogram,
    }
