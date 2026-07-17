import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from BLRun.runner import Runner


class Phase6OutputReductionTests(unittest.TestCase):
    def test_edge_list_compactor_keeps_strongest_per_target(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "Algorithms"
            / "compactEdgeList.awk"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "edges.tsv"
            output = root / "bounded.tsv"
            source.write_text(
                "Parent\tChild\tProbability\n"
                "a\tx\t0.1\n"
                "b\tx\t0.9\n"
                "c\tx\t0.8\n"
                "a\ty\t0.7\n"
                "b\ty\t0.2\n"
                "c\ty\t0.6\n",
                encoding="utf-8",
            )
            with output.open("w", encoding="utf-8") as output_file:
                subprocess.run(
                    [
                        "awk",
                        "-v", "top_k=2",
                        "-v", "input_header=1",
                        "-v", "source_col=1",
                        "-v", "target_col=2",
                        "-v", "score_col=3",
                        "-v", "output_source=Parent",
                        "-v", "output_target=Child",
                        "-v", "output_score=Probability",
                        "-F", "\t",
                        "-f", str(script),
                        str(source),
                    ],
                    check=True,
                    stdout=output_file,
                )
            bounded = pd.read_csv(output, sep="\t")

        self.assertEqual(len(bounded), 4)
        self.assertEqual(set(bounded[bounded.Child == "x"].Parent), {"b", "c"})
        self.assertEqual(set(bounded[bounded.Child == "y"].Parent), {"a", "c"})

    def test_rank_matrix_compactor_uses_grisli_score_conversion(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "Algorithms"
            / "compactRankMatrix.awk"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            genes = root / "genes.txt"
            matrix = root / "matrix.csv"
            output = root / "bounded.tsv"
            genes.write_text("a\nb\nc\n", encoding="utf-8")
            matrix.write_text("9,2,8\n1,7,6\n5,4,3\n", encoding="utf-8")
            with output.open("w", encoding="utf-8") as output_file:
                subprocess.run(
                    [
                        "awk",
                        "-v", "top_k=2",
                        "-v", f"gene_file={genes}",
                        "-f", str(script),
                        str(matrix),
                    ],
                    check=True,
                    stdout=output_file,
                )
            bounded = pd.read_csv(output, sep="\t")

        self.assertEqual(len(bounded), 6)
        target_a = bounded[bounded.Gene2 == "a"]
        self.assertEqual(set(target_a.Gene1), {"b", "c"})
        self.assertEqual(set(target_a.EdgeWeight), {8, 4})

    def test_edge_list_compactor_supports_grnvbem_six_column_output(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "Algorithms"
            / "compactEdgeList.awk"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "edges.tsv"
            output = root / "bounded.tsv"
            source.write_text(
                "Parent\tType\tChild\tWeight\tProbability\tScore\n"
                "a\t+\ttarget\t0.2\t0.3\t0.1\n"
                "b\t-\ttarget\t0.4\t0.9\t0.2\n"
                "c\t+\ttarget\t0.5\t0.7\t0.3\n",
                encoding="utf-8",
            )
            with output.open("w", encoding="utf-8") as output_file:
                subprocess.run(
                    [
                        "awk",
                        "-v", "top_k=2",
                        "-v", "input_header=1",
                        "-v", "source_col=1",
                        "-v", "target_col=3",
                        "-v", "score_col=5",
                        "-v", "output_source=Parent",
                        "-v", "output_target=Child",
                        "-v", "output_score=Probability",
                        "-F", "\t",
                        "-f", str(script),
                        str(source),
                    ],
                    check=True,
                    stdout=output_file,
                )
            bounded = pd.read_csv(output, sep="\t")

        self.assertEqual(list(bounded.Parent), ["b", "c"])
        self.assertEqual(list(bounded.Child), ["target", "target"])
        self.assertEqual(list(bounded.Probability), [0.9, 0.7])

    def test_bounded_trajectory_merge_matches_full_global_top_k(self):
        trajectories = [
            iter([("a", "x", 9), ("b", "x", 8), ("c", "x", 1)]),
            iter([("a", "x", 2), ("b", "x", 3), ("c", "x", 10)]),
        ]
        bounded = Runner._merge_bounded_trajectory_edges(trajectories, 2)

        self.assertEqual(set(bounded.Gene1), {"a", "c"})
        self.assertEqual(dict(zip(bounded.Gene1, bounded.EdgeWeight)), {"a": 9.0, "c": 10.0})


if __name__ == "__main__":
    unittest.main()
