import csv
import os
from pathlib import Path
import shlex
import pandas as pd
import numpy as np

from BLRun.runner import Runner


class LEAPRunner(Runner):
    """Concrete runner for the LEAP GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for LEAP.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        ExpressionData = self.read_expression_data()
        PTData = self.read_pseudotime_data()

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

    def run(self):
        '''
        Function to run LEAP algorithm

        Requires the maxLag parameter
        '''

        maxLag = str(self.params['maxLag'])
        top_k = str(self._resolve_top_k() or 0)
        script_path = (
            Path(__file__).resolve().parents[1]
            / 'Algorithms' / 'LEAP' / 'runLeap.R'
        )

        PTData = self.read_pseudotime_data()

        colNames = PTData.columns
        commands = []
        for idx in range(len(colNames)):
            cmdToRun = ' '.join(['docker run --rm',
                                f"-v {shlex.quote(str(self.working_dir))}:/usr/working_dir",
                                f"-v {shlex.quote(str(script_path))}:/runLeap.R:ro",
                                f'{self.image} /bin/sh -c \"time -v -o',
                                "/usr/working_dir/time" + str(idx) + ".txt",
                                'Rscript /runLeap.R',
                                "/usr/working_dir/ExpressionData" + str(idx) + ".csv",
                                maxLag,
                                "/usr/working_dir/outFile" + str(idx) + ".txt",
                                top_k, '\"'])

            commands.append(cmdToRun)

        self._run_docker_batch(commands)

    def _resolve_top_k(self):
        '''
        Resolve the maximum number of edges to keep per target gene.

        GRNScope keeps only the strongest ``maxRegulatorsPerTarget`` edges per
        target downstream, so retaining more here just materialises the full
        g x g edge list for nothing. Returns None when the cap is absent
        (standalone BEELINE), preserving the original full output.
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
        Function to parse outputs from LEAP.
        '''
        workDir = self.working_dir

        PTData = self.read_pseudotime_data()

        colNames = PTData.columns
        outFileNames = ['outFile' + str(indx) + '.txt' for indx in range(len(colNames))]

        # A missing trajectory must fail the run instead of publishing partial results.
        for outFileName in outFileNames:
            if not (workDir / outFileName).exists():
                raise FileNotFoundError(workDir / outFileName)

        top_k = self._resolve_top_k()

        if top_k is not None:
            merged = self._merge_bounded_trajectory_edges(
                (
                    self._iter_output_edges(workDir / out_file_name)
                    for out_file_name in outFileNames
                ),
                top_k,
            )
            self._write_ranked_edges(merged)
            return

        # General path: full parse across all trajectories (cross-trajectory max).
        self._parse_output_full(workDir, outFileNames)

    @staticmethod
    def _iter_output_edges(out_path):
        with out_path.open('r', newline='') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            for row in reader:
                try:
                    score = abs(float(row['Score']))
                except (KeyError, TypeError, ValueError):
                    continue
                yield row.get('Gene1', ''), row.get('Gene2', ''), score

    def _parse_output_topk(self, out_path, top_k):
        self._write_ranked_edges(self._merge_bounded_trajectory_edges(
            [self._iter_output_edges(out_path)], top_k))

    def _parse_output_full(self, workDir, colNames):
        self._write_ranked_edges(self._merge_trajectory_edges(
            self._iter_output_edges(workDir / ('outFile' + str(index) + '.txt'))
            for index in range(len(colNames))))
