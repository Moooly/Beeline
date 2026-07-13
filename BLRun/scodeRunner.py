import os
import pandas as pd
import numpy as np

from BLRun.runner import Runner


class SCODERunner(Runner):
    """Concrete runner for the SCODE GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for SCODE.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)
        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        for idx in range(len(colNames)):
            # Create output subdirectory in advance to prevent docker from
            # creating it with root-exclusive permissions
            (self.working_dir / str(idx)).mkdir(exist_ok=True)

            # Select cells belonging to each pseudotime trajectory
            colName = colNames[idx]
            index = PTData[colName].index[PTData[colName].notnull()]
            exprName = "ExpressionData"+str(idx)+".csv"
            ExpressionData.loc[:,index].to_csv(self.working_dir / exprName,
                                     sep = '\t', header  = False, index = False)
            cellName = "PseudoTime"+str(idx)+".csv"
            ptDF = PTData.loc[index,[colName]]
            # SCODE expects a column labeled PseudoTime.
            ptDF.rename(columns = {colName:'PseudoTime'}, inplace = True)
            # output file
            ptDF.to_csv(self.working_dir / cellName,
                                     sep = '\t', header  = False)

    def run(self):
        '''
        Function to run SCODE algorithm
        '''

        z = str(self.params['z'])
        nIter = str(self.params['nIter'])
        nRep = str(self.params['nRep'])

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns

        for idx in range(len(colNames)):

            ExpressionData = pd.read_csv(self.working_dir /
                                         ("ExpressionData"+str(idx)+".csv"),
                                         header = None, index_col = None, sep ='\t')
            nCells = str(ExpressionData.shape[1])
            nGenes = str(ExpressionData.shape[0])

            cmdToRun = ' '.join(['docker run --rm',
                                f'--user {os.getuid()}:{os.getgid()}',
                                '-e HOME=/tmp',
                                f"-v {self.working_dir}:/usr/working_dir",
                                f'{self.image} /bin/sh -c \"time -v -o',
                                "/usr/working_dir/time" + str(idx) + ".txt",
                                'ruby run_R.rb',
                                "/usr/working_dir/ExpressionData" + str(idx) + ".csv",
                                "/usr/working_dir/PseudoTime" + str(idx) + ".csv",
                                "/usr/working_dir/" + str(idx),
                                nGenes, z, nCells, nIter, nRep, '\"'])

            self._run_docker(cmdToRun, append=(idx > 0))

    def parseOutput(self):
        '''
        Function to parse outputs from SCODE.
        '''
        workDir = self.working_dir

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)
        colNames = PTData.columns

        # Quit if any trajectory output is missing (matches original behaviour).
        for indx in range(len(colNames)):
            if not (workDir / (str(indx)+'/meanA.txt')).exists():
                print(str(workDir / (str(indx)+'/meanA.txt')) + ' does not exist, skipping...')
                return

        ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                     header = 0, index_col = 0)
        GeneList = list(ExpressionData.index)

        top_k = self._resolve_top_k()

        # Bounded fast path: a single trajectory with a per-target cap. SCODE's
        # A matrix (meanA.txt) is inherently g x g, but the full g^2 edge list
        # need not be sorted/written: keep only the top-K regulators per target
        # (matrix column) via a partial sort. With one trajectory there is no
        # cross-trajectory max, so this is exact.
        if top_k is not None and len(colNames) == 1:
            self._parse_output_topk(workDir / '0/meanA.txt', GeneList, top_k)
            return

        # General path: original full sort / cross-trajectory max.
        self._parse_output_full(workDir, colNames, GeneList)

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

    def _parse_output_topk(self, meanA_path, GeneList, top_k):
        '''
        Keep only the top-K regulators per target (matrix column) by absolute
        weight, using np.argpartition so the full g^2 matrix is never sorted.
        '''
        OutMatrix = np.abs(pd.read_csv(meanA_path, sep='\t', header=None).values)
        n_rows, n_cols = OutMatrix.shape
        keep = min(top_k, n_rows)

        ranked_rows = []
        for col in range(n_cols):
            column = OutMatrix[:, col]
            if keep < n_rows:
                candidate_rows = np.argpartition(column, n_rows - keep)[n_rows - keep:]
            else:
                candidate_rows = np.arange(n_rows)
            # Order the kept regulators by descending absolute weight.
            candidate_rows = candidate_rows[np.argsort(column[candidate_rows])[::-1]]
            target = GeneList[col]
            for row in candidate_rows:
                ranked_rows.append((GeneList[row], target, column[row]))

        self._write_ranked_edges(
            pd.DataFrame(ranked_rows, columns=['Gene1', 'Gene2', 'EdgeWeight'])
        )

    def _parse_output_full(self, workDir, colNames, GeneList):
        '''
        Original full parse: sort every entry of each trajectory's A matrix,
        take the cross-trajectory max per edge, and rank descending.
        '''
        for indx in range(len(colNames)):
            OutDF = pd.read_csv(workDir / (str(indx)+'/meanA.txt'), sep = '\t', header = None)

            # Sort values in a matrix using code from:
            # https://stackoverflow.com/questions/21922806/sort-values-of-matrix-in-python
            OutMatrix = np.abs(OutDF.values)
            idx = np.argsort(OutMatrix, axis = None)[::-1]
            rows, cols = np.unravel_index(idx, OutDF.shape)
            DFSorted = OutMatrix[rows, cols]

            outFile = open(workDir / ('outFile'+str(indx)+'.csv'),'w')
            outFile.write('Gene1'+'\t'+'Gene2'+'\t'+'EdgeWeight'+'\n')

            for row, col, val in zip(rows, cols, DFSorted):
                outFile.write('\t'.join([GeneList[row],GeneList[col],str(val)])+'\n')
            outFile.close()

        OutSubDF = [0]*len(colNames)
        for indx in range(len(colNames)):
            outFile = 'outFile'+str(indx)+'.csv'
            OutSubDF[indx] = pd.read_csv(workDir / outFile, sep = '\t', header = 0)

            OutSubDF[indx].EdgeWeight = np.abs(OutSubDF[indx].EdgeWeight)

        outDF = pd.concat(OutSubDF)
        FinalDF = outDF[outDF['EdgeWeight'] == outDF.groupby(['Gene1','Gene2'])['EdgeWeight'].transform('max')]
        FinalDF = FinalDF.sort_values(['EdgeWeight'], ascending=False)
        self._write_ranked_edges(FinalDF[['Gene1', 'Gene2', 'EdgeWeight']])
