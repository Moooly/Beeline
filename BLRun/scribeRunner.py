import heapq
import os
import pandas as pd

from BLRun.runner import Runner


class SCRIBERunner(Runner):
    """Concrete runner for the SCRIBE GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for SCRIBE.
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
            ExpressionData.loc[:,index].to_csv(self.working_dir / exprName,
                                     sep = ',', header  = True, index = True)
            cellName = "pseudoTimeData"+str(idx)+".csv"
            ptDF = PTData.loc[index,[colName]]
            # Scribe expects a column labeled Time.
            ptDF.rename(columns = {colName:'Time'}, inplace = True)

            ptDF.to_csv(self.working_dir / cellName,
                                     sep = ',', header  = True, index = True)

        SCRIBE_GENE_FILE = self.working_dir / "GeneData.csv"
        if not SCRIBE_GENE_FILE.exists():
            # required column!!
            geneDict = {}
            geneDict['gene_short_name'] = [gene.replace('x_', '') for gene in ExpressionData.index]

            geneDF = pd.DataFrame(geneDict, index = ExpressionData.index)
            geneDF.to_csv(SCRIBE_GENE_FILE,
                          sep = ',', header = True)

    def run(self):
        '''
        Function to run SCRIBE algorithm.
        To see all the inputs runScribe.R script takes, run:
        docker run scribe:base /bin/sh -c "Rscript runScribe.R -h"
        '''

        # required inputs
        delay = str(self.params['delay'])
        method = str(self.params['method'])
        low = str(self.params['lowerDetectionLimit'])
        fam = str(self.params['expressionFamily'])

        # Build the command to run Scribe
        PTData = self.read_pseudotime_data()
        colNames = PTData.columns
        commands = []

        for idx in range(len(colNames)):
            # Specify file names for inputs and outputs
            exprName = "ExpressionData"+str(idx)+".csv"
            cellName = "pseudoTimeData"+str(idx)+".csv"
            outFile = "outFile"+str(idx)+".csv"
            timeFile = 'time'+str(idx)+".txt"

            cmdToRun = ' '.join(['docker run --rm',
                           f"-v {self.working_dir}:/usr/working_dir",
                           f'{self.image} /bin/sh -c \"time -v -o',
                           "/usr/working_dir/" + timeFile, 'Rscript runScribe.R',
                           '-e', "/usr/working_dir/" + exprName, '-c', "/usr/working_dir/" + cellName,
                           '-g', "/usr/working_dir/GeneData.csv", '-o /usr/working_dir/', '-d', delay, '-l', low,
                           '-m', method, '-x', fam, '--outFile ' + outFile])

            if str(self.params['log']) == 'True':
                cmdToRun += ' --log'
            if str(self.params['ignorePT']) == 'True':
                cmdToRun += ' -i'

            cmdToRun += '\"'

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
        Function to parse outputs from SCRIBE.
        '''
        workDir = self.working_dir

        PTData = self.read_pseudotime_data()
        colNames = PTData.columns

        # Quit if any trajectory output is missing (matches original behaviour).
        for idx in range(len(colNames)):
            if not (workDir / ('outFile'+str(idx)+'.csv')).exists():
                print(str(workDir / ('outFile'+str(idx)+'.csv')) + ' does not exist, skipping...')
                return

        top_k = self._resolve_top_k()

        # Bounded fast path: a single trajectory with a per-target cap. With one
        # trajectory there is no cross-trajectory max, so keeping only the top-K
        # edges per target (by |EdgeWeight|) in a heap is exact and never loads
        # the full g^2 directed edge list into memory.
        if top_k is not None and len(colNames) == 1:
            self._parse_output_topk(workDir / 'outFile0.csv', top_k)
            return

        # General path: original full parse across all trajectories.
        self._parse_output_full(workDir, colNames)

    def _parse_output_topk(self, out_path, top_k):
        '''
        Stream a single trajectory's space-separated edge list (Gene1 Gene2
        weight) and keep only the top-K edges per target (Gene2) by absolute
        weight in a heap, matching GRNScope's downstream per-target cap.
        '''
        target_heaps: dict = {}
        sequence = 0
        with out_path.open('r') as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    weight = float(parts[2])
                except ValueError:
                    continue
                gene1, gene2 = parts[0], parts[1]
                # GRNScope groups by Gene2 (target).
                heap = target_heaps.setdefault(gene2, [])
                item = (abs(weight), sequence, gene1, gene2, weight)
                sequence += 1
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif abs(weight) > heap[0][0]:
                    heapq.heapreplace(heap, item)

        ranked_rows = []
        for heap in target_heaps.values():
            for _abs_weight, _seq, gene1, gene2, weight in sorted(
                heap, key=lambda entry: (-entry[0], entry[1])
            ):
                ranked_rows.append((gene1, gene2, weight))

        self._write_ranked_edges(
            pd.DataFrame(ranked_rows, columns=['Gene1', 'Gene2', 'EdgeWeight'])
        )

    def _parse_output_full(self, workDir, colNames):
        '''
        Original full parse: cross-trajectory max per edge, ranked descending.
        '''
        OutSubDF = [0]*len(colNames)
        for idx in range(len(colNames)):
            OutSubDF[idx] = pd.read_csv(workDir / ('outFile'+str(idx)+'.csv'), sep = ' ', header = None)

        # megre the dataframe by taking the maximum value from each DF
        # From here: https://stackoverflow.com/questions/20383647/pandas-selecting-by-label-sometimes-return-series-sometimes-returns-dataframe
        outDF = pd.concat(OutSubDF)
        outDF.columns= ['Gene1','Gene2','EdgeWeight']
        # Group by rows code is from here:
        # https://stackoverflow.com/questions/53114609/pandas-how-to-remove-duplicate-rows-but-keep-all-rows-with-max-value
        res = outDF[outDF['EdgeWeight'] == outDF.groupby(['Gene1','Gene2'])['EdgeWeight'].transform('max')]
        # Sort values in the dataframe
        finalDF = res.sort_values('EdgeWeight',ascending=False)

        self._write_ranked_edges(finalDF[['Gene1', 'Gene2', 'EdgeWeight']])
