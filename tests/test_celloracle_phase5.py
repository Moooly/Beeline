import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from BLRun import celloracleRunner as celloracle_runner
from BLRun.celloracleRunner import CellOracleRunner


class FakeDockerWorker:
    instances = []

    def __init__(self, **settings):
        self.settings = settings
        self.stopped = False
        self.__class__.instances.append(self)

    def submit(self, runner):
        return {
            "status": "Completed",
            "worker_pid": 42,
            "request_index": 2,
            "prior_cache_hit": True,
            "preprocessing_mode": "oracle_native_with_metadata_embedding",
            "elapsed_seconds": 1.25,
            "edge_count": 11,
        }

    def stop(self):
        self.stopped = True


class CellOraclePhase5Tests(unittest.TestCase):
    def setUp(self):
        celloracle_runner.shutdown_celloracle_workers()
        FakeDockerWorker.instances.clear()

    def tearDown(self):
        celloracle_runner.shutdown_celloracle_workers()

    @staticmethod
    def make_runner(output_dir: Path) -> CellOracleRunner:
        runner = CellOracleRunner.__new__(CellOracleRunner)
        runner.image = "celloracle:test"
        runner.output_dir = output_dir
        runner.working_dir = output_dir / "working_dir"
        runner.working_dir.mkdir(parents=True)
        runner.params = {
            "species": "mouse",
            "baseGrn": "mouse_scATAC_atlas",
            "alpha": 7,
            "pValueCutoff": 0.01,
            "topK": 5,
            "maxGenes": 2000,
            "maxCells": 300,
        }
        runner.cpu_budget = 3
        runner.memory_budget_mb = 6000
        return runner

    def test_confidence_runs_share_one_scope_level_container(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "runtime" / "outputs"
            first = self.make_runner(
                output_root / "dataset" / "run-1" / "CELLORACLE"
            )
            second = self.make_runner(
                output_root / "dataset" / "run-2" / "CELLORACLE"
            )

            with patch.object(
                celloracle_runner,
                "_CellOracleDockerWorker",
                FakeDockerWorker,
            ):
                first_worker = celloracle_runner._get_celloracle_worker(first)
                second_worker = celloracle_runner._get_celloracle_worker(second)

            self.assertIs(first_worker, second_worker)
            self.assertEqual(len(FakeDockerWorker.instances), 1)
            self.assertEqual(first_worker.settings["cpu_budget"], 3)
            self.assertEqual(first_worker.settings["memory_budget_mb"], 6000)
            self.assertEqual(
                first_worker.settings["mount_root"],
                (Path(temp_dir) / "runtime").resolve(),
            )

    def test_runner_records_worker_and_prior_cache_telemetry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(
                Path(temp_dir)
                / "runtime"
                / "outputs"
                / "dataset"
                / "run-1"
                / "CELLORACLE"
            )
            worker = FakeDockerWorker(
                image=runner.image,
                mount_root=Path(temp_dir),
                cpu_budget=3,
                memory_budget_mb=6000,
            )
            with patch.object(
                celloracle_runner,
                "_get_celloracle_worker",
                return_value=worker,
            ):
                runner.run()

            self.assertEqual(
                runner.persistent_runtime_observability,
                {
                    "kind": "celloracle_container_worker",
                    "worker_pid": 42,
                    "request_index": 2,
                    "prior_cache_hit": True,
                    "preprocessing_mode": "oracle_native_with_metadata_embedding",
                    "worker_elapsed_seconds": 1.25,
                    "edge_count": 11,
                },
            )

    def test_worker_request_preserves_all_scientific_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir) / "output")
            arguments = runner._inference_arguments(
                in_file="/runtime/input.csv",
                out_file="/runtime/output.tsv",
            )

        joined = " ".join(arguments)
        self.assertIn("--species mouse", joined)
        self.assertIn("--baseGrn mouse_scATAC_atlas", joined)
        self.assertIn("--alpha 7", joined)
        self.assertIn("--pValueCutoff 0.01", joined)
        self.assertIn("--topK 5", joined)
        self.assertIn("--maxGenes 2000", joined)
        self.assertIn("--maxCells 300", joined)
        self.assertIn("--nJobs 3", joined)


if __name__ == "__main__":
    unittest.main()
