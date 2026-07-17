import shlex
import tempfile
import unittest
from pathlib import Path

from BLRun.genie3Runner import GENIE3Runner
from BLRun.grnboost2Runner import GRNBoost2Runner


class ArboretoRunnerCommandTests(unittest.TestCase):
    def _capture_command(self, runner_class, params):
        runner = runner_class.__new__(runner_class)
        runner.params = params
        runner.image = 'grnbeeline/arboreto:base'

        with tempfile.TemporaryDirectory(prefix='beeline path ') as temp_dir:
            runner.working_dir = Path(temp_dir)
            commands = []
            runner._run_docker = commands.append
            runner.run()

        self.assertEqual(len(commands), 1)
        return commands[0], Path(temp_dir)

    def test_genie3_mounts_matching_entrypoint_and_passes_parameters(self):
        command, working_dir = self._capture_command(
            GENIE3Runner,
            {'nEstimators': 250, 'maxFeatures': 'log2'},
        )

        self.assertIn('python /runArboreto.py --algo=GENIE3', command)
        self.assertIn('--nEstimators=250', command)
        self.assertIn('--maxFeatures=log2', command)
        self.assertIn('--topK=0', command)
        self.assertIn(':/runArboreto.py:ro', command)
        self.assertIn(
            f'-v {shlex.quote(f"{working_dir}:/usr/working_dir")}',
            command,
        )

    def test_grnboost2_mounts_matching_entrypoint_and_passes_parameters(self):
        command, _ = self._capture_command(
            GRNBoost2Runner,
            {
                'learningRate': 0.05,
                'nEstimators': 750,
                'maxFeatures': 0.25,
            },
        )

        self.assertIn('python /runArboreto.py --algo=GRNBoost2', command)
        self.assertIn('--learningRate=0.05', command)
        self.assertIn('--nEstimators=750', command)
        self.assertIn('--maxFeatures=0.25', command)
        self.assertIn('--topK=0', command)
        self.assertIn(':/runArboreto.py:ro', command)


if __name__ == '__main__':
    unittest.main()
