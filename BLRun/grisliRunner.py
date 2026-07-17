import csv
import os
from pathlib import Path
import shlex
import pandas as pd
import numpy as np

from BLRun.runner import Runner


class GRISLIRunner(Runner):
    """Concrete runner for the GRISLI GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for GRISLI.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        ExpressionData = self.read_expression_data()
        PTData = self.read_pseudotime_data()
        (self.working_dir / 'GeneList.txt').write_text(
            '\n'.join(str(gene) for gene in ExpressionData.index) + '\n',
            encoding='utf-8',
        )

        colNames = PTData.columns
        for idx in range(len(colNames)):
            (self.working_dir / str(idx)).mkdir(exist_ok = True)

            # Select cells belonging to each pseudotime trajectory
            colName = colNames[idx]
            index = PTData[colName].index[PTData[colName].notnull()]

            exprName = str(idx)+"/ExpressionData.tsv"
            ExpressionData.loc[:,index].to_csv(self.working_dir / exprName,
                                     sep = '\t', header  = False, index = False)

            cellName = str(idx)+"/PseudoTime.tsv"
            ptDF = PTData.loc[index,[colName]]
            ptDF.to_csv(self.working_dir / cellName,
                                     sep = '\t', header  = False, index = False)

    def run(self):
        '''
        Function to run GRISLI algorithm
        '''

        L = str(self.params['L'])
        R = str(self.params['R'])
        alphaMin = str(self.params['alphaMin'])
        top_k = self._resolve_top_k() or 0
        compactor_path = (
            Path(__file__).resolve().parents[1]
            / 'Algorithms' / 'compactRankMatrix.awk'
        )

        PTData = self.read_pseudotime_data()

        colNames = PTData.columns
        commands = []
        for idx in range(len(colNames)):
            os.makedirs(str(self.working_dir / str(idx)), exist_ok = True)
            output_path = (
                "/usr/working_dir/" + str(idx) + "/outFile.txt"
            )
            algorithm_output_path = (
                "/tmp/grisli-full.csv" if top_k else output_path
            )
            compact_command = (
                " && awk"
                f" -v top_k={top_k}"
                " -v gene_file=/usr/working_dir/GeneList.txt"
                f" -f /compactRankMatrix.awk {algorithm_output_path} > {output_path}"
                if top_k
                else ""
            )

            cmdToRun = ' '.join(['docker run --rm',
                                f"-v {shlex.quote(str(self.working_dir))}:/usr/working_dir",
                                f"-v {shlex.quote(str(compactor_path))}:/compactRankMatrix.awk:ro",
                                f'{self.image} /bin/sh -c \"time -v -o',
                                "/usr/working_dir/time" + str(idx) + ".txt",
                                './GRISLI',
                                "/usr/working_dir/" + str(idx) + "/",
                                algorithm_output_path,
                                L, R, alphaMin + compact_command, '\"'])

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
        Function to parse outputs from GRISLI.
        '''
        workDir = self.working_dir

        PTData = self.read_pseudotime_data()
        colNames = PTData.columns

        # Quit if any trajectory output is missing (matches original behaviour).
        for indx in range(len(colNames)):
            if not (workDir / (str(indx)+'/outFile.txt')).exists():
                print(str(workDir / (str(indx)+'/outFile.txt')) + ' does not exist, skipping...')
                return

        # read input file for list of gene names
        ExpressionData = self.read_expression_data()
        GeneList = list(ExpressionData.index)

        top_k = self._resolve_top_k()

        if top_k is not None:
            streams = (
                self._iter_matrix_topk(
                    workDir / (str(index) + '/outFile.txt'),
                    GeneList,
                    top_k,
                )
                for index in range(len(colNames))
            )
            self._write_ranked_edges(
                self._merge_bounded_trajectory_edges(streams, top_k)
            )
            return

        # General path: original full sort / cross-trajectory max.
        self._parse_output_full(workDir, colNames, GeneList)

    @staticmethod
    def _iter_matrix_topk(out_path, gene_list, top_k):
        with out_path.open('r', newline='') as handle:
            first_line = handle.readline()
        if first_line.startswith('Gene1\tGene2\tEdgeWeight'):
            with out_path.open('r', newline='') as handle:
                reader = csv.DictReader(handle, delimiter='\t')
                for row in reader:
                    yield row['Gene1'], row['Gene2'], float(row['EdgeWeight'])
            return
        matrix = pd.read_csv(out_path, sep=',', header=None).values
        n_rows, n_cols = matrix.shape
        total = len(gene_list) * len(gene_list)
        keep = min(top_k, n_rows)
        for col in range(n_cols):
            column = matrix[:, col]
            candidate_rows = np.argsort(column, kind='stable')[:keep]
            for row in candidate_rows:
                yield gene_list[row], gene_list[col], total - column[row]

    def _parse_output_topk(self, out_path, GeneList, top_k):
        '''
        Keep only the top-K regulators per target (matrix column) using a partial
        sort, so the full g^2 matrix is never fully sorted. GRISLI's strongest
        edges are the smallest matrix values (EdgeWeight = g^2 - value).
        '''
        OutMatrix = pd.read_csv(out_path, sep=',', header=None).values
        n_rows, n_cols = OutMatrix.shape
        total = len(GeneList) * len(GeneList)
        keep = min(top_k, n_rows)

        ranked_rows = []
        for col in range(n_cols):
            column = OutMatrix[:, col]
            if keep < n_rows:
                # smallest `keep` values == largest EdgeWeight (= total - value)
                candidate_rows = np.argpartition(column, keep - 1)[:keep]
            else:
                candidate_rows = np.arange(n_rows)
            # Order the kept regulators by descending EdgeWeight (ascending value).
            candidate_rows = candidate_rows[np.argsort(column[candidate_rows], kind='stable')]
            target = GeneList[col]
            for row in candidate_rows:
                ranked_rows.append((GeneList[row], target, total - column[row]))

        self._write_ranked_edges(
            pd.DataFrame(ranked_rows, columns=['Gene1', 'Gene2', 'EdgeWeight'])
        )

    def _parse_output_full(self, workDir, colNames, GeneList):
        '''
        Original full parse: sort every entry of each trajectory's matrix, take
        the cross-trajectory max per edge, and rank descending.
        '''
        OutSubDF = [0]*len(colNames)
        total = len(GeneList) * len(GeneList)

        for indx in range(len(colNames)):
            OutDF = pd.read_csv(workDir / (str(indx)+'/outFile.txt'), sep = ',', header = None)
            # Sort values in a matrix using code from:
            # https://stackoverflow.com/questions/21922806/sort-values-of-matrix-in-python
            OutMatrix = OutDF.values
            idx = np.argsort(OutMatrix, axis = None)
            rows, cols = np.unravel_index(idx, OutDF.shape)
            DFSorted = OutMatrix[rows, cols]

            outFileName = workDir / str(indx) / 'rankedEdges.csv'
            outFile = open(outFileName,'w')
            outFile.write('Gene1'+'\t'+'Gene2'+'\t'+'EdgeWeight'+'\n')

            for row, col, val in zip(rows, cols, DFSorted):
                outFile.write('\t'.join([GeneList[row],GeneList[col],str(total-val)])+'\n')
            outFile.close()

            OutSubDF[indx] = pd.read_csv(outFileName, sep = '\t', header = 0)

            # megre the dataframe by taking the maximum value from each DF
            # From here: https://stackoverflow.com/questions/20383647/pandas-selecting-by-label-sometimes-return-series-sometimes-returns-dataframe
        outDF = pd.concat(OutSubDF)

        res = outDF.groupby(['Gene1','Gene2'],as_index=False).max()
        # Sort values in the dataframe
        finalDF = res.sort_values('EdgeWeight',ascending=False)

        self._write_ranked_edges(finalDF)
