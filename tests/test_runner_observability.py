import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from BLRunner import collect_gnu_time_metrics, execute_runner, run_worker


GNU_TIME_OUTPUT = """\
        User time (seconds): 1.25
        System time (seconds): 0.25
        Percent of CPU this job got: 60%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:02.50
        Maximum resident set size (kbytes): 1234
"""


class FakeRunner:
    def __init__(self, root):
        self.input_dir = root / "input"
        self.output_dir = root / "output"
        self.working_dir = self.output_dir / "working_dir"
        self.input_dir.mkdir()
        self.working_dir.mkdir(parents=True)
        self.exprData = "ExpressionData.csv"
        self.pseudoTimeData = "PseudoTime.csv"
        self.image = "algorithm:test"
        self.params = {"parameter": 7}
        self._phase_setup_seconds = 0.125
        (self.input_dir / self.exprData).write_text("gene,cell\na,1\n")
        (self.input_dir / self.pseudoTimeData).write_text("cell,pt\ncell,0\n")

    def generateInputs(self):
        (self.working_dir / "generated.txt").write_text("generated")

    def run(self):
        (self.working_dir / "time.txt").write_text(GNU_TIME_OUTPUT)

    def parseOutput(self):
        (self.output_dir / "rankedEdges.csv").write_text(
            "Gene1\tGene2\tEdgeWeight\na\tb\t1\n"
        )


class RunnerObservabilityTests(unittest.TestCase):
    def test_collect_gnu_time_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir)
            (working_dir / "time.txt").write_text(GNU_TIME_OUTPUT)
            metrics = collect_gnu_time_metrics(working_dir)

        self.assertEqual(metrics["container_wall_seconds"], 2.5)
        self.assertEqual(metrics["cpu_total_seconds"], 1.5)
        self.assertEqual(metrics["effective_cpu_percent"], 60.0)
        self.assertEqual(metrics["maximum_resident_set_kib"], 1234)

    def test_execute_runner_writes_phase_timings_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner(Path(temp_dir))
            payload = execute_runner(runner)
            saved = json.loads(
                (runner.working_dir / "phase_timings.json").read_text()
            )

        self.assertEqual(payload["status"], "Completed")
        self.assertEqual(saved["stages_seconds"]["runner_setup"], 0.125)
        self.assertIn("input_generation", saved["stages_seconds"])
        self.assertIn("algorithm_execution", saved["stages_seconds"])
        self.assertIn("output_parsing", saved["stages_seconds"])
        self.assertEqual(saved["resource_usage"]["container_wall_seconds"], 2.5)
        self.assertIsNotNone(saved["fingerprints"]["expression_sha256"])
        self.assertIsNotNone(saved["fingerprints"]["ranked_edges_sha256"])

    def test_worker_processes_multiple_requests_without_restarting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ready_path = root / "ready.json"
            first_response = root / "first.json"
            second_response = root / "second.json"
            requests = "\n".join(
                json.dumps(
                    {
                        "request_id": request_id,
                        "config_path": str(root / f"{request_id}.yaml"),
                        "response_path": str(response_path),
                    }
                )
                for request_id, response_path in (
                    ("first", first_response),
                    ("second", second_response),
                )
            ) + "\n"

            with patch("BLRunner.sys.stdin", io.StringIO(requests)), patch(
                "BLRunner.execute_config"
            ) as execute_config_mock:
                run_worker(ready_path)

            ready = json.loads(ready_path.read_text())
            first = json.loads(first_response.read_text())
            second = json.loads(second_response.read_text())

        self.assertEqual(ready["status"], "Ready")
        self.assertEqual(first["status"], "Completed")
        self.assertEqual(second["status"], "Completed")
        self.assertEqual(execute_config_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
