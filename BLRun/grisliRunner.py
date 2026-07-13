import os
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

        ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)
        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

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

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        for idx in range(len(colNames)):
            os.makedirs(str(self.working_dir / str(idx)), exist_ok = True)

            cmdToRun = ' '.join(['docker run --rm',
                                f"-v {self.working_dir}:/usr/working_dir",
                                f'{self.image} /bin/sh -c \"time -v -o',
                                "/usr/working_dir/time" + str(idx) + ".txt",
                                './GRISLI',
                                "/usr/working_dir/" + str(idx) + "/",
                                "/usr/working_dir/" + str(idx) + "/outFile.txt",
                                L, R, alphaMin, '\"'])

            self._run_docker(cmdToRun, append=(idx > 0))

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

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)
        colNames = PTData.columns

        # Quit if any trajectory output is missing (matches original behaviour).
        for indx in range(len(colNames)):
            if not (workDir / (str(indx)+'/outFile.txt')).exists():
                print(str(workDir / (str(indx)+'/outFile.txt')) + ' does not exist, skipping...')
                return

        # read input file for list of gene names
        ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                     header = 0, index_col = 0)
        GeneList = list(ExpressionData.index)

        top_k = self._resolve_top_k()

        # Bounded fast path: a single trajectory with a per-target cap. GRISLI's
        # output is inherently a g x g rank matrix, but the full g^2 edge list
        # need not be sorted/written: keep only the top-K regulators per target
        # (matrix column). GRISLI encodes strength as EdgeWeight = g^2 - value,
        # so the strongest edges are the *smallest* matrix values. With one
        # trajectory there is no cross-trajectory max, so this is exact.
        if top_k is not None and len(colNames) == 1:
            self._parse_output_topk(workDir / '0/outFile.txt', GeneList, top_k)
            return

        # General path: original full sort / cross-trajectory max.
        self._parse_output_full(workDir, colNames, GeneList)

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
