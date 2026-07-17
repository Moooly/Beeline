import csv
import heapq
import os
from pathlib import Path
import shlex
import pandas as pd

from BLRun.runner import Runner


class SINCERITIESRunner(Runner):
    """Concrete runner for the SINCERITIES GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for SINCERITIES.
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
            newExpressionData = ExpressionData.loc[:,index].T
            # Perform quantile binning as recommeded in the paper
            # http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html#pandas.qcut
            nBins = int(self.params['nBins'])
            tQuantiles = pd.qcut(PTData.loc[index,colName], q = nBins, duplicates ='drop')
            mid = [(a.left + a.right)/2 for a in tQuantiles]

            newExpressionData['Time'] = mid
            newExpressionData.to_csv(self.working_dir / exprName,
                                 sep = ',', header  = True, index = False)

    def run(self):
        '''
        Function to run SINCERITIES algorithm
        '''

        PTData = self.read_pseudotime_data()
        top_k = str(self._resolve_top_k() or 0)
        script_path = (
            Path(__file__).resolve().parents[1]
            / 'Algorithms' / 'SINCERITIES' / 'runSinceritiesGRNScope.R'
        )

        colNames = PTData.columns
        commands = []
        for idx in range(len(colNames)):
            cmdToRun = ' '.join(['docker run --rm',
                                f"-v {shlex.quote(str(self.working_dir))}:/usr/working_dir",
                                f"-v {shlex.quote(str(script_path))}:/SINCERITIES/runSinceritiesGRNScope.R:ro",
                                f'{self.image} /bin/sh -c \"time -v -o',
                                "/usr/working_dir/time" + str(idx) + ".txt",
                                'Rscript runSinceritiesGRNScope.R',
                                "/usr/working_dir/ExpressionData" + str(idx) + ".csv",
                                "/usr/working_dir/outFile" + str(idx) + ".txt",
                                top_k, '\"'])

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
        Function to parse outputs from SINCERITIES.
        '''
        workDir = self.working_dir

        PTData = self.read_pseudotime_data()
        colNames = PTData.columns

        # Quit if any trajectory output is missing (matches original behaviour).
        for idx in range(len(colNames)):
            if not (workDir / ('outFile'+str(idx)+'.txt')).exists():
                print(str(workDir / ('outFile'+str(idx)+'.txt')) + ' does not exist, skipping...')
                return

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
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    interaction = abs(float(row['Interaction']))
                except (KeyError, TypeError, ValueError):
                    continue
                # Preserve the original SINCERITIES orientation correction.
                yield (
                    row.get('TargetGENES', ''),
                    row.get('SourceGENES', ''),
                    interaction,
                )

    def _parse_output_topk(self, out_path, top_k):
        '''
        Stream a single trajectory's edge list, keeping only the top-K edges per
        target in a heap. SINCERITIES' output orientation is relabelled so that
        GRNScope's target (Gene2) is SINCERITIES' SourceGENES, matching the
        column swap in the original full parse. Interaction is already absolute.
        '''
        target_heaps: dict = {}
        sequence = 0
        with out_path.open('r', newline='') as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                self._write_ranked_edges(
                    pd.DataFrame(columns=['Gene1', 'Gene2', 'EdgeWeight'])
                )
                return
            column = {name: index for index, name in enumerate(header)}
            try:
                source_col = column['SourceGENES']
                target_col = column['TargetGENES']
                interaction_col = column['Interaction']
            except KeyError as exc:
                raise ValueError(f"SINCERITIES output missing expected column: {exc}")

            widest_column = max(source_col, target_col, interaction_col)
            for row in reader:
                if len(row) <= widest_column:
                    continue
                try:
                    interaction = abs(float(row[interaction_col]))
                except ValueError:
                    continue
                source_gene = row[source_col]
                target_gene = row[target_col]
                # GRNScope's target is SINCERITIES' SourceGENES (relabelled Gene2).
                heap = target_heaps.setdefault(source_gene, [])
                item = (interaction, sequence, source_gene, target_gene)
                sequence += 1
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif interaction > heap[0][0]:
                    heapq.heapreplace(heap, item)

        ranked_rows = []
        for heap in target_heaps.values():
            for interaction, _seq, source_gene, target_gene in sorted(
                heap, key=lambda entry: (-entry[0], entry[1])
            ):
                # Relabel to GRNScope orientation: Gene1 = regulator (TargetGENES),
                # Gene2 = target (SourceGENES).
                ranked_rows.append((target_gene, source_gene, interaction))

        self._write_ranked_edges(
            pd.DataFrame(ranked_rows, columns=['Gene1', 'Gene2', 'EdgeWeight'])
        )

    def _parse_output_full(self, workDir, colNames):
        '''
        Original full parse: cross-trajectory max per edge, ranked descending,
        with SINCERITIES' output columns relabelled to GRNScope orientation.
        '''
        OutSubDF = [0]*len(colNames)
        for idx in range(len(colNames)):
            OutSubDF[idx] = pd.read_csv(workDir / ('outFile'+str(idx)+'.txt'), sep = ',', header = 0)

        # megre the dataframe by taking the maximum value from each DF
        # From here: https://stackoverflow.com/questions/20383647/pandas-selecting-by-label-sometimes-return-series-sometimes-returns-dataframe
        outDF = pd.concat(OutSubDF)
        # Group by rows code is from here:
        # https://stackoverflow.com/questions/53114609/pandas-how-to-remove-duplicate-rows-but-keep-all-rows-with-max-value
        res = outDF[outDF['Interaction'] == outDF.groupby(['SourceGENES','TargetGENES'])['Interaction'].transform('max')]
        # Sort values in the dataframe
        finalDF = res.sort_values('Interaction',ascending=False)
        finalDF.drop(labels = 'Edges',axis = 'columns', inplace = True)
        # SINCERITIES output is incorrectly orderd
        finalDF.columns = ['Gene2','Gene1','EdgeWeight']
        self._write_ranked_edges(finalDF[['Gene1', 'Gene2', 'EdgeWeight']])
