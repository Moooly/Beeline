import importlib.util
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

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
            "expression_mode": "normalized_log",
            "input_contract_version": 2,
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
            "expressionMode": "normalized_log",
            "inputContractVersion": 2,
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
                    "expression_mode": "normalized_log",
                    "input_contract_version": 2,
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
        self.assertIn("--expressionMode normalized_log", joined)
        self.assertIn("--inputContractVersion 2", joined)
        self.assertIn("--nJobs 3", joined)

    def test_expression_input_is_published_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir) / "output")
            expression = pd.DataFrame(
                [[1, 2], [3, 4]],
                index=["GeneA", "GeneB"],
                columns=["Cell1", "Cell2"],
            )
            runner.read_expression_data = lambda: expression

            runner.generateInputs()

            written = pd.read_csv(
                runner.working_dir / "ExpressionData.csv", index_col=0
            )
            pd.testing.assert_frame_equal(written, expression)
            self.assertEqual(list(runner.working_dir.glob(".*.tmp")), [])

    @staticmethod
    def load_in_container_script():
        script_path = (
            Path(celloracle_runner.__file__).resolve().parents[1]
            / "Algorithms"
            / "CELLORACLE"
            / "runCellOracle.py"
        )
        fake_anndata = types.ModuleType("anndata")
        fake_anndata.AnnData = object
        fake_celloracle = types.ModuleType("celloracle")
        fake_celloracle.Oracle = object
        fake_celloracle.data = types.SimpleNamespace(
            **{
                name: (lambda: None)
                for name in (
                    "load_human_promoter_base_GRN",
                    "load_mouse_promoter_base_GRN",
                    "load_rat_promoter_base_GRN",
                    "load_Pig_promoter_base_GRN",
                    "load_chicken_promoter_base_GRN",
                    "load_zebrafish_promoter_base_GRN",
                    "load_xenopus_tropicalis_promoter_base_GRN",
                    "load_drosophila_promoter_base_GRN",
                    "load_Celegans_promoter_base_GRN",
                    "load_Scerevisiae_promoter_base_GRN",
                )
            }
        )
        fake_scanpy = types.ModuleType("scanpy")
        fake_scanpy.pp = types.SimpleNamespace()
        module_name = "celloracle_in_container_test"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "anndata": fake_anndata,
                "celloracle": fake_celloracle,
                "scanpy": fake_scanpy,
            },
        ):
            spec.loader.exec_module(module)
        return module

    @staticmethod
    def load_perturbation_script():
        script_path = (
            Path(celloracle_runner.__file__).resolve().parents[1]
            / "Algorithms"
            / "CELLORACLE"
            / "runCellOraclePerturbation.py"
        )
        fake_anndata = types.ModuleType("anndata")
        fake_anndata.AnnData = object
        fake_celloracle = types.ModuleType("celloracle")
        fake_celloracle.Oracle = object
        fake_applications = types.ModuleType("celloracle.applications")
        fake_applications.Gradient_calculator = object
        fake_applications.Oracle_development_module = object
        fake_celloracle.applications = fake_applications
        fake_scanpy = types.ModuleType("scanpy")
        fake_distributions = types.ModuleType("perturbation_distributions")
        fake_distributions.summarize_expression_distribution = lambda **_: {}
        module_name = "celloracle_perturbation_test"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "anndata": fake_anndata,
                "celloracle": fake_celloracle,
                "celloracle.applications": fake_applications,
                "scanpy": fake_scanpy,
                "perturbation_distributions": fake_distributions,
            },
        ):
            spec.loader.exec_module(module)
        return module

    def test_development_field_uses_exact_scoring_flow_and_simulation_mask(self):
        module = self.load_perturbation_script()
        coordinates = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        reference_flow = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        )
        trajectory_perturbation_flow = np.asarray(
            [[2.0, 0.0], [0.0, -3.0], [4.0, 0.0], [0.0, -5.0]]
        )
        inner_product = np.sum(
            reference_flow * trajectory_perturbation_flow,
            axis=1,
        )
        gradient = types.SimpleNamespace(
            gridpoints_coordinates=coordinates,
            mass_filter=np.asarray([False, True, False, False]),
            ref_flow=reference_flow,
        )
        development = types.SimpleNamespace(
            gridpoints_coordinates=coordinates,
            mass_filter_reference=np.asarray([False, True, False, False]),
            mass_filter_simulation=np.asarray([False, False, True, False]),
            ref_flow=reference_flow,
            flow=trajectory_perturbation_flow,
            inner_product=inner_product,
        )

        field = module.extract_development_field(gradient, development)

        self.assertEqual(
            [(row["x"], row["y"]) for row in field["development_vectors"]],
            [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        )
        perturbation_rows = field["development_perturbation_vectors"]
        score_rows = field["inner_product_grid"]
        self.assertEqual(
            [(row["x"], row["y"]) for row in perturbation_rows],
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        )
        self.assertEqual(
            [(row["x"], row["y"]) for row in score_rows],
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        )

        reference_by_coordinate = {
            tuple(coordinates[index]): reference_flow[index]
            for index in range(len(coordinates))
        }
        for vector, score in zip(perturbation_rows, score_rows):
            coordinate = (vector["x"], vector["y"])
            expected_score = float(
                np.dot(
                    np.asarray([vector["dx"], vector["dy"]]),
                    reference_by_coordinate[coordinate],
                )
            )
            self.assertEqual(score["score"], expected_score)

    def test_network_import_keeps_normalized_log_values_unchanged(self):
        module = self.load_in_container_script()
        values = np.asarray([[0.0, np.log1p(3.0)]], dtype=np.float32)
        adata = types.SimpleNamespace(X=values.copy(), layers={})
        calls = []
        oracle = types.SimpleNamespace(
            import_anndata_as_raw_count=lambda **kwargs: calls.append(("raw", kwargs)),
            import_anndata_as_normalized_count=lambda **kwargs: calls.append(
                ("normalized", kwargs)
            ),
        )

        module.import_expression_into_oracle(
            oracle,
            adata,
            expression_mode="normalized_log",
            cluster_column_name="cluster",
            embedding_name="X_test",
        )

        self.assertEqual(calls[0][0], "normalized")
        np.testing.assert_array_equal(adata.X, values)
        np.testing.assert_allclose(adata.layers["raw_count"], [[0.0, 3.0]])

    def test_network_raw_import_delegates_the_single_log_transform_to_celloracle(self):
        module = self.load_in_container_script()
        values = np.asarray([[0.0, 3.0]], dtype=np.float32)
        adata = types.SimpleNamespace(X=values.copy(), layers={})
        calls = []
        oracle = types.SimpleNamespace(
            import_anndata_as_raw_count=lambda **kwargs: calls.append(("raw", kwargs)),
            import_anndata_as_normalized_count=lambda **kwargs: calls.append(
                ("normalized", kwargs)
            ),
        )

        module.import_expression_into_oracle(
            oracle,
            adata,
            expression_mode="raw_count",
            cluster_column_name="cluster",
            embedding_name="X_test",
        )

        self.assertEqual([call[0] for call in calls], ["raw"])
        np.testing.assert_array_equal(adata.X, values)

    def test_perturbation_import_uses_the_same_normalized_contract(self):
        module = self.load_perturbation_script()
        values = np.asarray([[np.log1p(2.0), np.log1p(5.0)]], dtype=np.float32)
        adata = types.SimpleNamespace(X=values.copy(), layers={})
        calls = []
        oracle = types.SimpleNamespace(
            import_anndata_as_raw_count=lambda **kwargs: calls.append(("raw", kwargs)),
            import_anndata_as_normalized_count=lambda **kwargs: calls.append(
                ("normalized", kwargs)
            ),
        )

        module.import_expression_into_oracle(
            oracle,
            adata,
            expression_mode="normalized_log",
            cluster_column_name="cluster",
            embedding_name="X_umap",
        )

        self.assertEqual([call[0] for call in calls], ["normalized"])
        np.testing.assert_array_equal(adata.X, values)
        np.testing.assert_allclose(
            adata.layers["raw_count"],
            [[2.0, 5.0]],
            rtol=1e-6,
        )

    def test_perturbation_embedding_does_not_retransform_normalized_log_input(self):
        module = self.load_perturbation_script()
        values = np.asarray([[0.0, np.log1p(4.0)]], dtype=np.float32)
        adata = types.SimpleNamespace(X=values.copy(), raw=None)
        calls = []
        module.sc.pp = types.SimpleNamespace(
            normalize_total=lambda *_args, **_kwargs: calls.append("normalize_total"),
            log1p=lambda *_args, **_kwargs: calls.append("log1p"),
            scale=lambda *_args, **_kwargs: calls.append("scale"),
        )

        model_input = module.prepare_embedding_expression(
            adata,
            "normalized_log",
        )

        self.assertEqual(calls, ["scale"])
        np.testing.assert_array_equal(model_input, values)

    def test_container_waits_for_delayed_bind_mount_contents(self):
        module = self.load_in_container_script()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "ExpressionData.csv"
            input_path.touch()
            expected = pd.DataFrame(
                [[1, 2], [3, 4]],
                index=["GeneA", "GeneB"],
                columns=["Cell1", "Cell2"],
            )

            def publish_contents():
                time.sleep(0.08)
                expected.to_csv(input_path)

            writer = threading.Thread(target=publish_contents)
            writer.start()
            try:
                observed = module.read_expression_when_ready(
                    input_path,
                    timeout_seconds=1.0,
                    poll_interval_seconds=0.01,
                )
            finally:
                writer.join()

            pd.testing.assert_frame_equal(observed, expected)


if __name__ == "__main__":
    unittest.main()
