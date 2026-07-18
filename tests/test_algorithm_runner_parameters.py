import tempfile
import unittest
from pathlib import Path

import pandas as pd

from BLRun.celloracleRunner import CellOracleRunner
from BLRun.grisliRunner import GRISLIRunner
from BLRun.grnvbemRunner import GRNVBEMRunner
from BLRun.leapRunner import LEAPRunner
from BLRun.pidcRunner import PIDCRunner
from BLRun.ppcorRunner import PPCORRunner
from BLRun.scodeRunner import SCODERunner
from BLRun.scribeRunner import SCRIBERunner
from BLRun.scsglRunner import SCSGLRunner
from BLRun.sinceritiesRunner import SINCERITIESRunner
from BLRun.singeRunner import SINGERunner


class AlgorithmRunnerParameterTests(unittest.TestCase):
    def test_singe_local_image_contains_runtime_commands_used_by_runner(self):
        dockerfile = (
            Path(__file__).resolve().parents[1]
            / "Algorithms"
            / "SINGE"
            / "Dockerfile"
        ).read_text(encoding="utf-8")

        self.assertIn("time", dockerfile)
        self.assertIn("octave", dockerfile)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="beeline-parameter-test-")
        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.working_dir = self.output_dir / "working_dir"
        self.input_dir.mkdir()
        self.working_dir.mkdir(parents=True)

        cells = [f"cell-{index}" for index in range(8)]
        self.expression = pd.DataFrame(
            {cell: [index + 1, index + 2, index + 3] for index, cell in enumerate(cells)},
            index=["gene-a", "gene-b", "gene-c"],
        )
        self.pseudotime = pd.DataFrame(
            {"trajectory": range(len(cells))},
            index=cells,
        )
        self.expression.to_csv(self.input_dir / "ExpressionData.csv")
        self.pseudotime.to_csv(self.input_dir / "PseudoTime.csv")

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_runner(self, runner_class, params):
        runner = runner_class.__new__(runner_class)
        runner.params = dict(params)
        runner.image = "algorithm:test"
        runner.input_dir = self.input_dir
        runner.output_dir = self.output_dir
        runner.working_dir = self.working_dir
        runner.exprData = "ExpressionData.csv"
        runner.pseudoTimeData = "PseudoTime.csv"
        commands = []
        runner._run_docker = (
            lambda command, append=False: commands.append((command, append))
        )
        return runner, commands

    def test_celloracle_forwards_all_exposed_parameters(self):
        runner, commands = self.make_runner(
            CellOracleRunner,
            {"maxCells": 123, "pValueCutoff": 0.2},
        )
        runner.run()

        command = commands[0][0]
        self.assertIn("--maxCells 123", command)
        self.assertIn("--pValueCutoff 0.2", command)

    def test_scode_forwards_all_exposed_parameters(self):
        runner, commands = self.make_runner(
            SCODERunner,
            {"z": 2, "nIter": 7, "nRep": 4},
        )
        self.expression.to_csv(
            self.working_dir / "ExpressionData0.csv",
            sep="\t",
            header=False,
            index=False,
        )
        runner.run()

        command = commands[0][0]
        self.assertIn(" 3 2 8 7 4 ", command)

    def test_scribe_forwards_numeric_choice_and_boolean_parameters(self):
        runner, commands = self.make_runner(
            SCRIBERunner,
            {
                "delay": 2,
                "method": "RDI",
                "lowerDetectionLimit": 0.1,
                "expressionFamily": "negbinomial",
                "log": True,
                "ignorePT": False,
            },
        )
        runner.run()

        command = commands[0][0]
        self.assertIn("-d 2", command)
        self.assertIn("-l 0.1", command)
        self.assertIn("-m RDI", command)
        self.assertIn("-x negbinomial", command)
        self.assertIn("--log", command)
        self.assertNotIn(" -i", command)

    def test_singe_writes_all_exposed_parameters_and_replicate_count(self):
        runner, commands = self.make_runner(
            SINGERunner,
            {
                "lambda": 0.2,
                "dT": 4,
                "num_lags": 3,
                "kernel_width": 1.5,
                "prob_zero_removal": 0.2,
                "prob_remove_samples": 0.3,
                "family": "poisson",
                "num_replicates": 2,
            },
        )
        runner.run()

        lines = (self.working_dir / "hyperparameters.txt").read_text().splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertIn("--lambda 0.2", line)
            self.assertIn("--dT 4", line)
            self.assertIn("--num-lags 3", line)
            self.assertIn("--kernel-width 1.5", line)
            self.assertIn("--prob-zero-removal 0.2", line)
            self.assertIn("--prob-remove-samples 0.3", line)
            self.assertIn("--family poisson", line)
        self.assertEqual(len(commands), 1)
        self.assertIn("fullKp(1, 12)", commands[0][0])

    def test_leap_and_grisli_forward_their_parameters(self):
        leap, leap_commands = self.make_runner(LEAPRunner, {"maxLag": 0.2})
        leap.run()
        self.assertIn(" 0.2 ", leap_commands[0][0])

        grisli, grisli_commands = self.make_runner(
            GRISLIRunner,
            {
                "L": 2,
                "R": 3,
                "alphaMin": 0.2,
                "maxRegulatorsPerTarget": 2,
            },
        )
        grisli.run()
        self.assertIn(" 2 3 0.2 ", grisli_commands[0][0])
        self.assertIn("/tmp/grisli-full.csv", grisli_commands[0][0])
        self.assertIn("compactRankMatrix.awk", grisli_commands[0][0])

    def test_phase6_entrypoints_receive_the_output_limit(self):
        pidc, pidc_commands = self.make_runner(
            PIDCRunner,
            {"maxRegulatorsPerTarget": 4},
        )
        pidc.run()
        self.assertIn("julia /runPIDC.jl", pidc_commands[0][0])
        self.assertIn("outFile.txt 4", pidc_commands[0][0])

        grnvbem, grnvbem_commands = self.make_runner(
            GRNVBEMRunner,
            {"maxRegulatorsPerTarget": 4},
        )
        grnvbem.run()
        self.assertIn("/tmp/grnvbem-full.txt", grnvbem_commands[0][0])
        self.assertIn("compactEdgeList.awk", grnvbem_commands[0][0])
        self.assertIn(
            "source_col=1 -v target_col=3 -v score_col=5",
            grnvbem_commands[0][0],
        )

        ppcor, ppcor_commands = self.make_runner(
            PPCORRunner,
            {"pVal": 0.05, "maxRegulatorsPerTarget": 4},
        )
        ppcor.run()
        self.assertIn("Rscript /runPPCOR.R", ppcor_commands[0][0])
        self.assertIn(" 0.05 4", ppcor_commands[0][0])

    def test_scsgl_forwards_density_and_association_parameters(self):
        runner, commands = self.make_runner(
            SCSGLRunner,
            {"pos_density": 0.2, "neg_density": 0.3, "assoc": "dotprod"},
        )
        runner.run()
        command = commands[0][0]
        self.assertIn("--pos_density=0.2", command)
        self.assertIn("--neg_density=0.3", command)
        self.assertIn("--assoc=dotprod", command)

    def test_sincerities_uses_the_selected_bin_count(self):
        runner, _commands = self.make_runner(SINCERITIESRunner, {"nBins": 2})
        runner.generateInputs()

        generated = pd.read_csv(self.working_dir / "ExpressionData0.csv")
        self.assertEqual(generated["Time"].nunique(), 2)

    def test_ppcor_uses_the_selected_p_value_cutoff(self):
        runner, _commands = self.make_runner(PPCORRunner, {"pVal": 0.05})
        (self.working_dir / "outFile.txt").write_text(
            "Gene1\tGene2\tcorVal\tpValue\n"
            "gene-a\tgene-c\t0.9\t0.10\n"
            "gene-b\tgene-c\t-0.8\t0.01\n",
            encoding="utf-8",
        )
        runner.parseOutput()

        ranked = pd.read_csv(self.output_dir / "rankedEdges.csv", sep="\t")
        weights = dict(zip(ranked["Gene1"], ranked["EdgeWeight"]))
        self.assertEqual(weights["gene-a"], 0.0)
        self.assertEqual(weights["gene-b"], -0.8)


if __name__ == "__main__":
    unittest.main()
