import csv
import heapq
import os
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
        Function to run GRN-VBEM algorithm
        '''

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        for idx in range(len(colNames)):
            cmdToRun = ' '.join(['docker run --rm',
                                f"-v {self.working_dir}:/usr/working_dir",
                                f'{self.image} /bin/sh -c \"time -v -o',
                                "/usr/working_dir/time" + str(idx) + ".txt",
                                './GRNVBEM',
                                "/usr/working_dir/ExpressionData" + str(idx) + ".csv",
                                "/usr/working_dir/outFile" + str(idx) + ".txt", '\"'])

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
        Function to parse outputs from GRNVBEM.
        '''
        workDir = self.working_dir

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns

        # Quit if any trajectory output is missing (matches original behaviour).
        for indx in range(len(colNames)):
            if not (workDir / ('outFile'+str(indx)+'.txt')).exists():
                print(str(workDir / ('outFile'+str(indx)+'.txt')) + ' does not exist, skipping...')
                return

        top_k = self._resolve_top_k()

        # Bounded fast path: a single trajectory with a per-target cap. GRNVBEM
        # scores every parent->child pair (g^2 edges); with one trajectory there
        # is no cross-trajectory max, so keeping only the top-K parents per target
        # (Child) in a heap is exact and never loads the full g^2 list into memory.
        if top_k is not None and len(colNames) == 1:
            self._parse_output_topk(workDir / 'outFile0.txt', top_k)
            return

        # General path: original full parse across all trajectories.
        self._parse_output_full(workDir, colNames)

    def _parse_output_topk(self, out_path, top_k):
        '''
        Stream a single trajectory's edge list, keeping only the top-K edges per
        target (Child) by probability in a heap. GRNScope's target is the Child
        gene; the emitted orientation matches the original column rename
        (Parent -> Gene1, Child -> Gene2, Probability -> EdgeWeight).
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
                parent_col = column['Parent']
                child_col = column['Child']
                probability_col = column['Probability']
            except KeyError as exc:
                raise ValueError(f"GRNVBEM output missing expected column: {exc}")

            widest_column = max(parent_col, child_col, probability_col)
            for row in reader:
                if len(row) <= widest_column:
                    continue
                try:
                    probability = float(row[probability_col])
                except ValueError:
                    continue
                parent = row[parent_col]
                child = row[child_col]
                # GRNScope's target is the Child gene; keep its strongest parents.
                heap = target_heaps.setdefault(child, [])
                item = (abs(probability), sequence, parent, child, probability)
                sequence += 1
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif abs(probability) > heap[0][0]:
                    heapq.heapreplace(heap, item)

        ranked_rows = []
        for heap in target_heaps.values():
            for _abs_probability, _seq, parent, child, probability in sorted(
                heap, key=lambda entry: (-entry[0], entry[1])
            ):
                ranked_rows.append((parent, child, probability))

        self._write_ranked_edges(
            pd.DataFrame(ranked_rows, columns=['Gene1', 'Gene2', 'EdgeWeight'])
        )

    def _parse_output_full(self, workDir, colNames):
        '''
        Original full parse: cross-trajectory max per edge, ranked descending,
        with GRNVBEM's columns renamed to GRNScope orientation.
        '''
        OutSubDF = [0]*len(colNames)
        for indx in range(len(colNames)):
            OutSubDF[indx] = pd.read_csv(workDir / ('outFile'+str(indx)+'.txt'), sep = '\t', header = 0)

        outDF = pd.concat(OutSubDF)
        FinalDF = outDF[outDF['Probability'] == outDF.groupby(['Parent','Child'])['Probability'].transform('max')]

        self._write_ranked_edges(
            FinalDF.sort_values('Probability', ascending=False).rename(
                columns={'Parent': 'Gene1', 'Child': 'Gene2', 'Probability': 'EdgeWeight'}
            )[['Gene1', 'Gene2', 'EdgeWeight']]
        )
