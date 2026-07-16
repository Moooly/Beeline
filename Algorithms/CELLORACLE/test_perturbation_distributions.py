from __future__ import annotations

import unittest

from perturbation_distributions import summarize_expression_distribution


class PerturbationDistributionTests(unittest.TestCase):
    def test_uses_shared_bins_and_preserves_cell_totals(self):
        summary = summarize_expression_distribution(
            gene="GATA1",
            baseline_values=[0, 1, 2, 3],
            simulated_values=[0, 2, 4, 6],
            scope_type="global",
            bin_count=3,
        )

        self.assertEqual(summary["gene"], "GATA1")
        self.assertEqual(summary["cell_count"], 4)
        self.assertEqual(summary["baseline_mean"], 1.5)
        self.assertEqual(summary["simulated_mean"], 3.0)
        self.assertEqual(summary["increased_cell_fraction"], 0.75)
        self.assertEqual(
            sum(bin_["baseline_count"] for bin_ in summary["histogram"]),
            4,
        )
        self.assertEqual(
            sum(bin_["simulated_count"] for bin_ in summary["histogram"]),
            4,
        )

    def test_constant_values_create_one_readable_bin(self):
        summary = summarize_expression_distribution(
            gene="HBB",
            baseline_values=[2, 2, 2],
            simulated_values=[2, 2, 2],
            scope_type="cluster",
            scope_label="Cluster A",
        )

        self.assertEqual(len(summary["histogram"]), 1)
        self.assertEqual(summary["histogram"][0]["start"], 2)
        self.assertEqual(summary["histogram"][0]["end"], 2)
        self.assertEqual(summary["increased_cell_fraction"], 0)
        self.assertEqual(summary["decreased_cell_fraction"], 0)

    def test_filters_non_finite_pairs(self):
        summary = summarize_expression_distribution(
            gene="SOX9",
            baseline_values=[1, float("nan"), 3],
            simulated_values=[2, 4, float("inf")],
            scope_type="global",
        )

        self.assertEqual(summary["cell_count"], 1)
        self.assertEqual(summary["mean_change"], 1)


if __name__ == "__main__":
    unittest.main()
