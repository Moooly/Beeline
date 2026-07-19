import csv
import heapq
import os
import pandas as pd

from BLRun.runner import Runner


class SINGERunner(Runner):
    """Concrete runner for the SINGE GRN inference algorithm."""

    # A 1,999-gene / 818-cell GLG process used about 3.1 GiB in production.
    # Reserve additional headroom for MATLAB runtime variance and Docker.
    _REPLICATE_MEMORY_MB = 3584
    _MEMORY_SAFETY_FRACTION = 0.85

    def generateInputs(self):
        '''
        Function to generate desired inputs for SINGE.
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
            newExpressionData['PseudoTime'] = PTData.loc[index,colName]
            newExpressionData.to_csv(self.working_dir / exprName,
                                 sep = ',', header  = True, index = False)

    def run(self):
        '''Function to run SINGE algorithm.'''

        params_order = [
            'lambda', 'dT', 'num_lags', 'kernel_width',
            'prob_zero_removal', 'prob_remove_samples',
            'family',
        ]
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
        replicates = [
            ' '.join(
                '--' + parameter.replace('_', '-') + ' ' + str(params[parameter])
                for parameter in params_order
            )
            + ' '.join(['', '--replicate', str(replicate), '--ID', str(replicate)])
            for replicate in range(num_replicates)
        ]
        (self.working_dir / 'hyperparameters.txt').write_text(
            '\n'.join(replicates) + '\n',
            encoding='utf-8',
        )

        PTData = self.read_pseudotime_data()
        trajectory_inputs = []
        prepare_commands = []
        for idx in range(len(PTData.columns)):
            os.makedirs(str(self.working_dir / str(idx)), exist_ok=True)
            out_file_symlink = 'out' + str(idx)
            input_file = '/usr/working_dir/ExpressionData' + str(idx) + '.csv'
            input_mat = '/usr/working_dir/ExpressionData' + str(idx) + '.mat'
            gene_list_mat = '/usr/working_dir/GeneList' + str(idx) + '.mat'
            params_file = '/usr/working_dir/hyperparameters.txt'

            convert_input_to_matfile = 'octave -q --eval \\"CSV = csvread(\'' + input_file + '\'); ' + \
                'X = sparse(CSV(2:end,1:end-1).\'); ptime = CSV(2:end,end).\'; ' + \
                'Kp2.Kp = single(ptime); Kp2.sumKp = single(ptime*X.\'); fullKp(1, ' + \
                str(int(params['dT']) * int(params['num_lags'])) + ') = Kp2; ' + \
                'save(\'-v7\',\'' + input_mat + '\', \'X\', \'ptime\', \'fullKp\'); ' + \
                'f = fopen(\'' + input_file + '\'); gene_list = strsplit(fgetl(f), \',\')(1:end-1).\'; fclose(f); ' + \
                'save(\'-v7\',\'' + gene_list_mat + '\', \'gene_list\')\\"'

            prepare_commands.append(' '.join([
                'docker run --rm --entrypoint /bin/sh',
                f'-v {self.working_dir}:/usr/working_dir',
                f'{self.image} -c "',
                convert_input_to_matfile,
                '"',
            ]))
            trajectory_inputs.append(
                (idx, input_mat, gene_list_mat, out_file_symlink, params_file)
            )

        worker_limit = self._resolve_parallel_worker_limit(
            max(1, len(trajectory_inputs) * num_replicates)
        )
        self._run_docker_batch(
            prepare_commands,
            max_workers=worker_limit,
            log_prefix='singe-prepare',
        )

        replicate_commands = []
        for idx, input_mat, gene_list_mat, out_file_symlink, params_file in trajectory_inputs:
            symlink_output = ' '.join([
                'ln -s',
                '/usr/working_dir/' + str(idx) + '/',
                out_file_symlink,
            ])
            for replicate in range(num_replicates):
                replicate_commands.append(' '.join([
                    'docker run --rm --entrypoint /bin/sh',
                    f'-v {self.working_dir}:/usr/working_dir',
                    f'{self.image} -c "',
                    symlink_output,
                    '&& time -v -o',
                    f'/usr/working_dir/time{idx}-replicate-{replicate}.txt',
                    '/usr/local/SINGE/SINGE.sh /usr/local/MATLAB/MATLAB_Runtime/v94 GLG',
                    input_mat,
                    gene_list_mat,
                    out_file_symlink,
                    params_file,
                    str(replicate + 1),
                    '"',
                ]))

        self._run_docker_batch(
            replicate_commands,
            append=True,
            max_workers=worker_limit,
            log_prefix='singe-replicate',
        )

        aggregate_commands = []
        for idx, input_mat, gene_list_mat, out_file_symlink, _params_file in trajectory_inputs:
            symlink_output = ' '.join([
                'ln -s',
                '/usr/working_dir/' + str(idx) + '/',
                out_file_symlink,
            ])
            aggregate_commands.append(' '.join([
                'docker run --rm --entrypoint /bin/sh',
                f'-v {self.working_dir}:/usr/working_dir',
                f'{self.image} -c "',
                symlink_output,
                '&& time -v -o',
                f'/usr/working_dir/time{idx}-aggregate.txt',
                '/usr/local/SINGE/SINGE.sh /usr/local/MATLAB/MATLAB_Runtime/v94 Aggregate',
                input_mat,
                gene_list_mat,
                out_file_symlink,
                '"',
            ]))

        self._run_docker_batch(
            aggregate_commands,
            append=True,
            max_workers=worker_limit,
            log_prefix='singe-aggregate',
        )

    def _resolve_parallel_worker_limit(self, command_count):
        cpu_budget = max(1, int(getattr(self, 'cpu_budget', 1)))
        trajectory_workers = max(
            1,
            int(getattr(self, 'trajectory_workers', cpu_budget)),
        )
        worker_limit = min(max(1, int(command_count)), cpu_budget, trajectory_workers)
        memory_budget_mb = getattr(self, 'memory_budget_mb', None)
        if memory_budget_mb is not None:
            safe_memory_mb = int(
                max(512, int(memory_budget_mb)) * self._MEMORY_SAFETY_FRACTION
            )
            worker_limit = min(
                worker_limit,
                max(1, safe_memory_mb // self._REPLICATE_MEMORY_MB),
            )
        return max(1, worker_limit)

    def parseOutput(self):
        '''
        Function to parse outputs from SINGE.
        '''
        workDir = self.working_dir

        PTData = self.read_pseudotime_data()

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
