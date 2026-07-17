import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from BLRun import runner as runner_module
from BLRun.runner import Runner


class MinimalRunner(Runner):
    def generateInputs(self):
        pass

    def run(self):
        pass

    def parseOutput(self):
        pass


class RunnerInputCacheTests(unittest.TestCase):
    def test_source_matrix_is_parsed_once_and_sliced_for_each_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expression_source = root / "expression.csv"
            pseudotime_source = root / "pseudotime.csv"
            expression_source.write_text(
                ",cell-a,cell-b,cell-c\n"
                "gene-a,1,2,3\n"
                "gene-b,4,5,6\n",
                encoding="utf-8",
            )
            pseudotime_source.write_text(
                ",trajectory\n"
                "cell-a,0.1\n"
                "cell-b,0.2\n"
                "cell-c,0.3\n",
                encoding="utf-8",
            )
            runner_module._DATAFRAME_CACHE.clear()

            real_read_csv = runner_module.pd.read_csv
            with patch.object(
                runner_module.pd,
                "read_csv",
                wraps=real_read_csv,
            ) as read_csv_mock:
                observed_columns = []
                observed_rows = []
                for run_id, cells in (
                    ("run-1", ["cell-a", "cell-b"]),
                    ("run-2", ["cell-b", "cell-c"]),
                ):
                    selection_path = root / f"{run_id}.json"
                    selection_path.write_text(json.dumps(cells), encoding="utf-8")
                    instance = MinimalRunner.__new__(MinimalRunner)
                    instance.input_dir = root
                    instance.exprData = "unused-expression.csv"
                    instance.pseudoTimeData = "unused-pseudotime.csv"
                    instance.expression_source = expression_source
                    instance.pseudotime_source = pseudotime_source
                    instance.selected_cells_file = selection_path
                    instance._selected_cells_cache = None

                    observed_columns.append(
                        list(instance.read_expression_data().columns)
                    )
                    observed_rows.append(
                        list(instance.read_pseudotime_data().index)
                    )

            self.assertEqual(read_csv_mock.call_count, 2)
            self.assertEqual(
                observed_columns,
                [["cell-a", "cell-b"], ["cell-b", "cell-c"]],
            )
            self.assertEqual(
                observed_rows,
                [["cell-a", "cell-b"], ["cell-b", "cell-c"]],
            )


if __name__ == "__main__":
    unittest.main()
