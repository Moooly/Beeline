import shlex
import csv
import os
from pathlib import Path
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
        if num_replicates < 1:
            raise ValueError('SINGE requires at least one replicate.')
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
                f'-v {shlex.quote(str(self.working_dir))}:/usr/working_dir',
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

        # One short-lived container per allocated slot. Sequential tasks in
        # that slot reuse the MATLAB extraction cache without a long-lived
        # worker service or reduced replicate-level concurrency.
        slot_tasks = [[] for _ in range(min(worker_limit, len(trajectory_inputs) * num_replicates))]
        task_index = 0
        for idx, input_mat, gene_list_mat, out_file_symlink, params_file in trajectory_inputs:
            for replicate in range(num_replicates):
                slot_tasks[task_index % len(slot_tasks)].append(' '.join([
                    'ln -sfn', '/usr/working_dir/' + str(idx) + '/', out_file_symlink,
                    '&& time -v -o', f'/usr/working_dir/time{idx}-replicate-{replicate}.txt',
                    '/usr/local/SINGE/SINGE.sh /usr/local/MATLAB/MATLAB_Runtime/v94 GLG',
                    input_mat, gene_list_mat, out_file_symlink, params_file, str(replicate + 1),
                ]))
                task_index += 1
        wrapper = Path(__file__).resolve().parents[1] / 'Algorithms/SINGE/SINGE.sh'
        replicate_commands = [' '.join([
            'docker run --rm --entrypoint /bin/sh',
            f'-v {shlex.quote(str(self.working_dir))}:/usr/working_dir',
            f'-v {shlex.quote(str(wrapper))}:/usr/local/SINGE/SINGE.sh:ro',
            f'{self.image} -c', shlex.quote(' && '.join(tasks)),
        ]) for tasks in slot_tasks]
        self._run_docker_batch(
            replicate_commands, append=True, max_workers=worker_limit,
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
                f'-v {shlex.quote(str(self.working_dir))}:/usr/working_dir',
                f'-v {shlex.quote(str(wrapper))}:/usr/local/SINGE/SINGE.sh:ro',
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

        # A missing trajectory must fail the run instead of publishing partial results.
        for idx in range(len(colNames)):
            if not (workDir / str(idx) / 'SINGE_Ranked_Edge_List.txt').exists():
                raise FileNotFoundError(workDir / str(idx) / 'SINGE_Ranked_Edge_List.txt')

        from BLRun.edges import reduce_edges
        streams = (self._iter_edges(workDir / str(index) / 'SINGE_Ranked_Edge_List.txt') for index in range(len(colNames)))
        self._write_ranked_edges(reduce_edges(streams, self._resolve_top_k()))

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

    @staticmethod
    def _iter_edges(path):
        with path.open('r') as handle:
            next(handle, None)
            for line in handle:
                parts = line.split('\t')
                if len(parts) != 3:
                    raise ValueError("Malformed edge row in " + str(path))
                yield parts[0], parts[1], float(parts[2])

    def _parse_output_topk(self, out_path, top_k):
        from BLRun.edges import reduce_edges
        self._write_ranked_edges(reduce_edges([self._iter_edges(out_path)], top_k))

    def _parse_output_full(self, workDir, colNames):
        from BLRun.edges import reduce_edges
        self._write_ranked_edges(reduce_edges(
            (self._iter_edges(workDir / str(index) / 'SINGE_Ranked_Edge_List.txt') for index in range(len(colNames)))))
