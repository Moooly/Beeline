import shlex
import os
from pathlib import Path
import pandas as pd

from BLRun.runner import Runner


class SCRIBERunner(Runner):
    """Concrete runner for the SCRIBE GRN inference algorithm."""

    _MEMORY_PER_PAIR_WORKER_MB = 768
    _MIN_CONTAINER_MEMORY_MB = 768

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
        expression = self.read_expression_data()
        pair_workers = self._resolve_pair_workers(
            len(colNames), gene_count=len(expression),
            cell_count=int(PTData.notna().sum().max()),
        )
        script_path = Path(__file__).resolve().parents[1] / 'Algorithms/SCRIBE/runScribe.R'
        commands = []

        for idx in range(len(colNames)):
            # Specify file names for inputs and outputs
            exprName = "ExpressionData"+str(idx)+".csv"
            cellName = "pseudoTimeData"+str(idx)+".csv"
            outFile = "outFile"+str(idx)+".csv"
            timeFile = 'time'+str(idx)+".txt"

            cmdToRun = ' '.join(['docker run --rm',
                           f"-v {shlex.quote(str(self.working_dir))}:/usr/working_dir",
                           f"-v {shlex.quote(str(script_path))}:/runScribe.R:ro",
                           f'{self.image} /bin/sh -c \"time -v -o',
                           "/usr/working_dir/" + timeFile, 'Rscript /runScribe.R',
                           '-e', "/usr/working_dir/" + exprName, '-c', "/usr/working_dir/" + cellName,
                           '-g', "/usr/working_dir/GeneData.csv", '-o /usr/working_dir/', '-d', delay, '-l', low,
                           '-m', method, '-x', fam, '-w', str(pair_workers),
                           '--outFile ' + outFile])

            if str(self.params['log']) == 'True':
                cmdToRun += ' --log'
            if str(self.params['ignorePT']) == 'True':
                cmdToRun += ' -i'

            cmdToRun += '\"'

            commands.append(cmdToRun)

        self._run_docker_batch(commands)

    @staticmethod
    def _estimate_pair_memory_bytes(gene_count, cell_count, delay_count, workers):
        # Upstream allocates dense results even for a subset of gene pairs.
        # Include parent merge/serialization, pair graphs and expression copies;
        # preserve headroom for the conditioned stage and the R runtime.
        dense_result = (8 * delay_count + 12) * gene_count ** 2
        shared = 2 * dense_result + 48 * gene_count ** 2 + 32 * gene_count * cell_count
        worker = dense_result + 16 * gene_count ** 2 + 24 * gene_count * cell_count
        return 384 * 1024 ** 2 + shared + workers * worker

    def _resolve_pair_workers(self, trajectory_count, *, gene_count=None, cell_count=None):
        trajectory_count = max(1, int(trajectory_count))
        cpu = max(1, int(getattr(self, 'cpu_budget', 1)))
        trajectories = self._trajectory_worker_count(trajectory_count)
        cpu_workers = max(1, cpu // trajectories)
        memory = getattr(self, 'memory_budget_mb', None)
        if memory is None:
            return cpu_workers
        allowance = max(512, int(memory) // trajectories) * 1024 ** 2
        if gene_count is None:
            expression = self.read_expression_data()
            gene_count, cell_count = expression.shape
        delay_count = len(str(self.params.get('delay', 1)).split(','))
        # The fixed floor still protects runtime startup on small inputs.
        limit = min(cpu_workers, max(1, int(allowance // (768 * 1024 ** 2))))
        for workers in range(limit, 0, -1):
            if self._estimate_pair_memory_bytes(gene_count, cell_count, delay_count, workers) <= allowance * 0.8:
                return workers
        raise RuntimeError(
            'SCRIBE cannot fit the estimated dense pair matrices in its memory budget. '
            "Lower SCRIBE's Gene filter or trajectory concurrency, or allocate more memory."
        )

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

        # A missing trajectory must fail the run instead of publishing partial results.
        for idx in range(len(colNames)):
            if not (workDir / ('outFile'+str(idx)+'.csv')).exists():
                raise FileNotFoundError(workDir / ('outFile'+str(idx)+'.csv'))

        from BLRun.edges import reduce_edges
        streams = (self._iter_edges(workDir / ('outFile' + str(index) + '.csv')) for index in range(len(colNames)))
        self._write_ranked_edges(reduce_edges(streams, self._resolve_top_k()))

    @staticmethod
    def _iter_edges(path):
        with path.open('r') as handle:
            # SCRIBE emits a headerless whitespace-separated table.
            for line in handle:
                parts = line.split(None)
                if len(parts) != 3:
                    raise ValueError("Malformed edge row in " + str(path))
                yield parts[0], parts[1], float(parts[2])

    def _parse_output_topk(self, out_path, top_k):
        from BLRun.edges import reduce_edges
        self._write_ranked_edges(reduce_edges([self._iter_edges(out_path)], top_k))

    def _parse_output_full(self, workDir, colNames):
        from BLRun.edges import reduce_edges
        self._write_ranked_edges(reduce_edges(
            (self._iter_edges(workDir / ('outFile' + str(index) + '.csv')) for index in range(len(colNames)))))
