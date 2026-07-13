import csv
import heapq
import os
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

        ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)
        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

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

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        for idx in range(len(colNames)):
            cmdToRun = ' '.join(['docker run --rm',
                                f"-v {self.working_dir}:/usr/working_dir",
                                f'{self.image} /bin/sh -c \"time -v -o',
                                "/usr/working_dir/time" + str(idx) + ".txt",
                                'Rscript runLeap.R',
                                "/usr/working_dir/ExpressionData" + str(idx) + ".csv",
                                maxLag,
                                "/usr/working_dir/outFile" + str(idx) + ".txt", '\"'])

            self._run_docker(cmdToRun, append=(idx > 0))

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

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        outFileNames = ['outFile' + str(indx) + '.txt' for indx in range(len(colNames))]

        # Quit if any trajectory output is missing (matches original behaviour).
        for outFileName in outFileNames:
            if not (workDir / outFileName).exists():
                print(str(workDir / outFileName) + ' does not exist, skipping...')
                return

        top_k = self._resolve_top_k()

        # Bounded fast path: a single trajectory with a per-target cap. With one
        # trajectory there is no cross-trajectory max, so keeping only the top-K
        # edges per target (by |Score|) in a heap is exact and never loads the
        # full g^2 edge list into memory.
        if top_k is not None and len(colNames) == 1:
            self._parse_output_topk(workDir / outFileNames[0], top_k)
            return

        # General path: full parse across all trajectories (cross-trajectory max).
        self._parse_output_full(workDir, outFileNames)

    def _parse_output_topk(self, out_path, top_k):
        '''
        Stream a single trajectory's edge list and keep only the top-K edges per
        target (by absolute correlation) using a heap, so the full g^2 list is
        never held in memory. LEAP is unsigned, so the weight is |Score|.
        '''
        target_heaps: dict = {}
        sequence = 0
        with out_path.open('r', newline='') as handle:
            reader = csv.reader(handle, delimiter='\t')
            header = next(reader, None)
            if header is None:
                self._write_ranked_edges(
                    pd.DataFrame(columns=['Gene1', 'Gene2', 'EdgeWeight'])
                )
                return
            column = {name: index for index, name in enumerate(header)}
            try:
                gene1_col, gene2_col, score_col = (
                    column['Gene1'], column['Gene2'], column['Score']
                )
            except KeyError as exc:
                raise ValueError(f"LEAP output missing expected column: {exc}")

            widest_column = max(gene1_col, gene2_col, score_col)
            for row in reader:
                if len(row) <= widest_column:
                    continue
                try:
                    score = abs(float(row[score_col]))
                except ValueError:
                    continue
                gene1, gene2 = row[gene1_col], row[gene2_col]
                heap = target_heaps.setdefault(gene2, [])
                item = (score, sequence, gene1, gene2)
                sequence += 1
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, item)

        ranked_rows = []
        for heap in target_heaps.values():
            for score, _seq, gene1, gene2 in sorted(
                heap, key=lambda entry: (-entry[0], entry[1])
            ):
                ranked_rows.append((gene1, gene2, score))

        self._write_ranked_edges(
            pd.DataFrame(ranked_rows, columns=['Gene1', 'Gene2', 'EdgeWeight'])
        )

    def _parse_output_full(self, workDir, outFileNames):
        '''
        Original full parse: absolute correlation, max across trajectories per
        edge, ranked descending.
        '''
        OutSubDF = [0] * len(outFileNames)
        for indx, outFileName in enumerate(outFileNames):
            OutSubDF[indx] = pd.read_csv(workDir / outFileName, sep='\t', header=0)
            OutSubDF[indx].Score = np.abs(OutSubDF[indx].Score)
        outDF = pd.concat(OutSubDF)
        FinalDF = outDF[outDF['Score'] == outDF.groupby(['Gene1', 'Gene2'])['Score'].transform('max')]

        self._write_ranked_edges(FinalDF.sort_values('Score', ascending=False).rename(
            columns={'Score': 'EdgeWeight'}
        )[['Gene1', 'Gene2', 'EdgeWeight']])
