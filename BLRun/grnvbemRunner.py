import csv
import math
from pathlib import Path
import shlex
import pandas as pd

from BLRun.runner import Runner


class GRNVBEMRunner(Runner):
    """Concrete runner for the GRN-VBEM GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for GRNVBEM.
        It will create the input folder at self.working_dir if it
        does not exist already. The input folder will contain an ExpressionData.csv with
        cells ordered according to the pseudotime along the columns, and genes along
        the rows. If the files already exist, this function will overwrite it.
        '''

        ExpressionData = self.read_expression_data()
        PTData = self.read_pseudotime_data()
        max_genes = self._resolve_max_genes()
        if max_genes is not None and len(ExpressionData.index) > max_genes:
            # GRNScope applies this cap once before confidence subsampling. This
            # runner-side guard keeps standalone BEELINE use consistent with the
            # exposed parameter while preserving deterministic variance ties.
            variances = ExpressionData.var(axis=1, ddof=0)
            retained_genes = (
                variances.sort_values(ascending=False, kind='mergesort')
                .head(max_genes)
                .index
            )
            ExpressionData = ExpressionData.loc[retained_genes]

        colNames = PTData.columns
        for idx in range(len(colNames)):
            # Select cells belonging to each pseudotime trajectory
            colName = colNames[idx]
            index = PTData[colName].index[PTData[colName].notnull()]
            exprName = "ExpressionData"+str(idx)+".csv"

            subPT = PTData.loc[index,:]
            subExpr = ExpressionData[index]
            # Order columns by PseudoTime
            newExpressionData = subExpr[subPT.sort_values([colName]).index.astype(str)]

            newExpressionData.insert(loc = 0, column = 'GENES', \
                                                         value = newExpressionData.index)

            # Write .csv file
            newExpressionData.to_csv(self.working_dir / exprName,
                                 sep = ',', header  = True, index = False)

    def _resolve_max_genes(self):
        raw = self.params.get('maxGenes')
        if raw is None:
            return None
        try:
            max_genes = int(raw)
        except (TypeError, ValueError):
            return None
        return max_genes if max_genes >= 3 else None

    def run(self):
        '''
        Function to run GRN-VBEM algorithm
        '''

        PTData = self.read_pseudotime_data()
        top_k = self._resolve_top_k() or 0
        compactor_path = (
            Path(__file__).resolve().parents[1]
            / 'Algorithms' / 'compactEdgeList.awk'
        )

        colNames = PTData.columns
        commands = []
        for idx in range(len(colNames)):
            output_path = "/usr/working_dir/outFile" + str(idx) + ".txt"
            algorithm_output_path = (
                "/tmp/grnvbem-full.txt" if top_k else output_path
            )
            compact_command = (
                " && awk"
                f" -v top_k={top_k} -v input_header=1 -v strict_grnvbem=1"
                " -v source_col=1 -v target_col=3 -v score_col=6"
                " -v output_source=Parent -v output_target=Child"
                " -v output_score=Score -F '\\t'"
                f" -f /compactEdgeList.awk {algorithm_output_path} > {output_path}"
                if top_k
                else ""
            )
            cmdToRun = ' '.join(['docker run --rm',
                                f"-v {shlex.quote(str(self.working_dir))}:/usr/working_dir",
                                f"-v {shlex.quote(str(compactor_path))}:/compactEdgeList.awk:ro",
                                f'{self.image} /bin/sh -c \"time -v -o',
                                "/usr/working_dir/time" + str(idx) + ".txt",
                                './GRNVBEM',
                                "/usr/working_dir/ExpressionData" + str(idx) + ".csv",
                                algorithm_output_path + compact_command, '\"'])

            commands.append(cmdToRun)

        self._run_docker_batch(commands)

    def _resolve_top_k(self):
        '''
        Resolve the maximum number of edges to keep per target gene. GRNScope
        keeps only the strongest ``maxRegulatorsPerTarget`` edges per target
        downstream, so retaining more just materialises the full g^2 edge list
        for nothing. Returns None when absent (standalone BEELINE).
        '''
        raw = self.params.get('maxRegulatorsPerTarget')
        if raw is None:
            return None
        try:
            top_k = int(raw)
        except (TypeError, ValueError):
            return None
        return top_k if top_k > 0 else None

    def parseOutput(self):
        '''
        Function to parse outputs from GRNVBEM.
        '''
        workDir = self.working_dir

        PTData = self.read_pseudotime_data()

        colNames = PTData.columns

        # Quit if any trajectory output is missing (matches original behaviour).
        for indx in range(len(colNames)):
            if not (workDir / ('outFile'+str(indx)+'.txt')).exists():
                raise FileNotFoundError(workDir / ('outFile'+str(indx)+'.txt'))

        top_k = self._resolve_top_k()

        if top_k is not None:
            merged = self._merge_bounded_trajectory_edges(
                (
                    self._iter_output_edges(
                        workDir / ('outFile' + str(idx) + '.txt')
                    )
                    for idx in range(len(colNames))
                ),
                top_k,
            )
            self._write_ranked_edges(merged)
            return

        # General path: original full parse across all trajectories.
        self._parse_output_full(workDir, colNames)

    @staticmethod
    def _iter_output_edges(out_path):
        with out_path.open('r', newline='') as handle:
            reader = csv.reader(handle, delimiter='\t')
            header = next(reader, None)
            if not header:
                raise ValueError(f'GRNVBEM output is missing its header: {out_path}')

            # The bundled GRNVBEM binary writes a space-delimited header above
            # tab-delimited six-column rows. The bounded compactor writes a
            # conventional three-column tab-delimited header instead.
            if 'Score' in header:
                parent_col = header.index('Parent')
                child_col = header.index('Child')
                score_col = header.index('Score')
            elif header[0].split() == [
                'Parent', '-', 'Child', 'Weight', 'Probability', 'Score'
            ]:
                parent_col, child_col, score_col = 0, 2, 5
            else:
                raise ValueError(
                    "GRNVBEM output header does not expose Parent, Child, and "
                    f"Score columns: {header!r}."
                )

            widest_column = max(parent_col, child_col, score_col)
            for line_number, row in enumerate(reader, start=2):
                if len(row) <= widest_column or not row[parent_col] or not row[child_col]:
                    raise ValueError(f'Malformed GRNVBEM row {line_number} in {out_path}')
                try:
                    score = float(row[score_col])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'Invalid GRNVBEM score on row {line_number}') from exc
                if not math.isfinite(score):
                    raise ValueError(f'Nonfinite GRNVBEM score on row {line_number}')
                yield (
                    row[parent_col],
                    row[child_col],
                    score,
                )

    def _parse_output_topk(self, out_path, top_k):
        '''
        Stream a single trajectory's edge list, keeping only the top-K edges per
        target (Child) by signed-score magnitude. GRNScope's target is the Child
        gene; the emitted orientation matches the original column rename
        (Parent -> Gene1, Child -> Gene2, Score -> EdgeWeight).
        '''
        self._write_ranked_edges(
            self._merge_bounded_trajectory_edges(
                (self._iter_output_edges(out_path),),
                top_k,
            )
        )

    def _parse_output_full(self, workDir, colNames):
        '''
        Use GRNVBEM's signed Score rather than its unsigned posterior
        Probability, retaining the strongest score magnitude across
        trajectories.
        '''
        merged = self._merge_trajectory_edges(
            self._iter_output_edges(
                workDir / ('outFile' + str(index) + '.txt')
            )
            for index in range(len(colNames))
        )
        self._write_ranked_edges(merged)
