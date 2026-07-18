import tempfile
import threading
import unittest
from pathlib import Path

import pandas as pd

from BLRun.celloracleRunner import _CellOracleDockerWorker
from BLRun.runner import Runner


class MinimalRunner(Runner):
    def generateInputs(self):
        pass

    def run(self):
        pass

    def parseOutput(self):
        pass


class RunnerResourceTests(unittest.TestCase):
    def test_python_39_compatible_annotations_are_postponed(self):
        self.assertEqual(
            Runner._apply_docker_resource_limits.__annotations__["cpu_budget"],
            "int | None",
        )
        self.assertEqual(
            _CellOracleDockerWorker.__init__.__annotations__["memory_budget_mb"],
            "int | None",
        )

    def make_runner(self, root):
        runner = MinimalRunner.__new__(MinimalRunner)
        runner.output_dir = root
        runner.cpu_budget = 4
        runner.memory_budget_mb = 8000
        runner.trajectory_workers = 2
        return runner

    def test_docker_command_receives_cpu_memory_and_thread_limits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            command = runner._apply_docker_resource_limits(
                "docker run --rm image command"
            )

        self.assertIn("--cpus=4", command)
        self.assertIn("--memory=8000m", command)
        self.assertIn("OMP_NUM_THREADS=4", command)
        self.assertIn("OPENBLAS_NUM_THREADS=4", command)
        self.assertIn("JULIA_NUM_THREADS=4", command)

    def test_trajectory_batch_splits_budget_between_workers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runner = self.make_runner(root)
            calls = []
            calls_lock = threading.Lock()

            def fake_run(command, **kwargs):
                with calls_lock:
                    calls.append((command, kwargs))
                kwargs["output_path"].write_text(command, encoding="utf-8")

            runner._run_docker = fake_run
            runner._run_docker_batch(["first", "second", "third"])

            combined_output = (root / "output.txt").read_text(encoding="utf-8")

        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call[1]["cpu_budget"] == 2 for call in calls))
        self.assertTrue(all(call[1]["memory_budget_mb"] == 4000 for call in calls))
        self.assertEqual(combined_output, "firstsecondthird")

    def test_ranked_edges_file_is_preserved_and_bounded_per_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runner = self.make_runner(root)
            runner.params = {"maxRegulatorsPerTarget": 2}
            runner._write_ranked_edges(
                pd.DataFrame(
                    [
                        ("a", "target-1", 0.1),
                        ("b", "target-1", -0.8),
                        ("c", "target-1", 0.7),
                        ("a", "target-2", 0.4),
                        ("b", "target-2", 0.2),
                        ("c", "target-2", -0.5),
                        ("invalid", "target-2", float("nan")),
                    ],
                    columns=["Gene1", "Gene2", "EdgeWeight"],
                )
            )
            ranked_path = root / "rankedEdges.csv"
            ranked = pd.read_csv(ranked_path, sep="\t")

        self.assertTrue(ranked_path.name == "rankedEdges.csv")
        self.assertEqual(len(ranked), 4)
        self.assertEqual(
            set(ranked[ranked.Gene2 == "target-1"].Gene1),
            {"b", "c"},
        )
        self.assertEqual(
            set(ranked[ranked.Gene2 == "target-2"].Gene1),
            {"a", "c"},
        )
        self.assertTrue(
            runner.output_reduction_observability["ranked_edges_preserved"]
        )


if __name__ == "__main__":
    unittest.main()
