import csv
import heapq
import os
import pandas as pd

from BLRun.runner import Runner


class SINGERunner(Runner):
    """Concrete runner for the SINGE GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for SINGE.
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
            newExpressionData = ExpressionData.loc[:,index].T
            newExpressionData['PseudoTime'] = PTData.loc[index,colName]
            newExpressionData.to_csv(self.working_dir / exprName,
                                 sep = ',', header  = True, index = False)

    def run(self):
        '''
        Function to run SINGE algorithm
        '''

        # if the parameters aren't specified, then use default parameters
        # TODO allow passing in multiple sets of hyperparameters
        # these must be in the right order!
        params_order = [
            'lambda', 'dT', 'num_lags', 'kernel_width',
            'prob_zero_removal', 'prob_remove_samples',
            'family'
        ]
        # Standalone-BEELINE fallbacks, used only when a param is omitted from
        # the config. Kept in sync with the GRNScope algorithm registry
        # (backend/app/algorithm_registry.py, SINGE) so both entry points share
        # one set of defaults. GRNScope always supplies these explicitly.
        default_params = {
            'lambda': '0.01',
            'dT': '15',
            'num_lags': '5',
            'kernel_width': '0.5',
            'prob_zero_removal': '0',
            'prob_remove_samples': '0.0',
            'family': 'gaussian',
            'num_replicates': '3',
        }
        params = self.params
        for param, val in default_params.items():
            if param not in params:
                params[param] = val

        num_replicates = int(params['num_replicates'])
        replicates = []
        for replicate in range(num_replicates):
           replicates.append(' '.join('--' + p.replace('_', '-') + ' ' + str(params[p]) for p in params_order) + ' '.join(['', '--replicate', str(replicate), '--ID', str(replicate)]))
        params_str = '\n'.join(replicates)

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        for idx in range(len(colNames)):
            os.makedirs(str(self.working_dir / str(idx)), exist_ok = True)

            outFileSymlink = "out" + str(idx)
            inputFile = "/usr/working_dir/ExpressionData"+str(idx)+".csv"
            inputMat = "/usr/working_dir/ExpressionData"+str(idx)+".mat"
            geneListMat = "/usr/working_dir/GeneList"+str(idx)+".mat"
            paramsFile = "/usr/working_dir/hyperparameters.txt"

            '''
            This is a workaround for https://github.com/gitter-lab/SINGE/blob/master/code/parseParams.m#L39
            not allowing '/' characters in the outDir parameter.
            '''
            symlink_out_file = ' '.join(['ln -s', "/usr/working_dir/" + str(idx) + "/", outFileSymlink])

            '''
            See https://github.com/gitter-lab/SINGE/blob/master/README.md.  SINGE expects a data matfile with variables "X" and "ptime",
            and a gene_list matfile with the variable "gene_list".

            Saving fullKp is a very hacky workaround for https://github.com/gitter-lab/SINGE/blob/master/code/iLasso_for_SINGE.m#L56,
            that assumes this input was saved in matfile v7.3 which octave does not support.
            '''
            convert_input_to_matfile = 'octave -q --eval \\"CSV = csvread(\'' + inputFile + '\'); ' + \
                                 'X = sparse(CSV(2:end,1:end-1).\'); ptime = CSV(2:end,end).\'; ' + \
                                 'Kp2.Kp = single(ptime); Kp2.sumKp = single(ptime*X.\'); fullKp(1, ' + \
                                 str(int(params['dT'])*int(params['num_lags'])) + ') = Kp2; ' + \
                                 'save(\'-v7\',\'' + inputMat + '\', \'X\', \'ptime\', \'fullKp\'); ' + \
                                 'f = fopen(\'' + inputFile + '\'); gene_list = strsplit(fgetl(f), \',\')(1:end-1).\'; fclose(f); ' + \
                                 'save(\'-v7\',\'' + geneListMat + '\', \'gene_list\')\\"'

            cmdToRun = ' '.join(['docker run --rm --entrypoint /bin/sh',
                                f"-v {self.working_dir}:/usr/working_dir",
                                f'{self.image} -c \"echo \\"',
                                 params_str, '\\" >', paramsFile, '&&', symlink_out_file, '&&', convert_input_to_matfile,
                                 '&& time -v -o', "/usr/working_dir/time" + str(idx) + ".txt",
                                 '/usr/local/SINGE/SINGE.sh /usr/local/MATLAB/MATLAB_Runtime/v94 standalone',
                                 inputMat, geneListMat, outFileSymlink, paramsFile, '\"'])

            self._run_docker(cmdToRun, append=(idx > 0))

    def parseOutput(self):
        '''
        Function to parse outputs from SINGE.
        '''
        workDir = self.working_dir

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns

        # Quit if any trajectory output is missing (matches original behaviour).
        for idx in range(len(colNames)):
            if not (workDir / str(idx) / 'SINGE_Ranked_Edge_List.txt').exists():
                print(str(workDir / str(idx) / 'SINGE_Ranked_Edge_List.txt') + ' does not exist, skipping...')
                return

        top_k = self._resolve_top_k()

        # Bounded fast path: a single trajectory with a per-target cap. With one
        # trajectory there is no cross-trajectory max, so keeping only the top-K
        # edges per target (by |EdgeWeight|) in a heap is exact and never loads
        # the full g^2 edge list into memory.
        if top_k is not None and len(colNames) == 1:
            self._parse_output_topk(workDir / '0' / 'SINGE_Ranked_Edge_List.txt', top_k)
            return

        # General path: original full parse across all trajectories.
        self._parse_output_full(workDir, colNames)

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

    def _parse_output_topk(self, out_path, top_k):
        '''
        Stream a single trajectory's ranked edge list (tab-separated, with a
        header) and keep only the top-K edges per target (Gene2, column 1) by
        absolute weight in a heap, matching GRNScope's downstream per-target cap.
        '''
        target_heaps: dict = {}
        sequence = 0
        with out_path.open('r', newline='') as handle:
            reader = csv.reader(handle, delimiter='\t')
            next(reader, None)  # discard header row (column names are overwritten)
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    weight = float(row[2])
                except ValueError:
                    continue
                gene1, gene2 = row[0], row[1]
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
            OutSubDF[idx] = pd.read_csv(workDir / str(idx) / 'SINGE_Ranked_Edge_List.txt',
                                sep = '\t', header = 0)
        # megre the dataframe by taking the maximum value from each DF
        # Code from here:
        # https://stackoverflow.com/questions/20383647/pandas-selecting-by-label-sometimes-return-series-sometimes-returns-dataframe
        outDF = pd.concat(OutSubDF)
        outDF.columns= ['Gene1','Gene2','EdgeWeight']
        # Group by rows code is from here:
        # https://stackoverflow.com/questions/53114609/pandas-how-to-remove-duplicate-rows-but-keep-all-rows-with-max-value
        res = outDF[outDF['EdgeWeight'] == outDF.groupby(['Gene1','Gene2'])['EdgeWeight'].transform('max')]
        # Sort values in the dataframe
        finalDF = res.sort_values('EdgeWeight', ascending=False)
        self._write_ranked_edges(finalDF[['Gene1', 'Gene2', 'EdgeWeight']])
