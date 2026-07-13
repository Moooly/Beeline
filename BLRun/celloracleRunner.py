import csv
import heapq
import os
import pandas as pd

from BLRun.runner import Runner


class CellOracleRunner(Runner):
    """Concrete runner for the CellOracle GRN inference algorithm.

    CellOracle is prior-informed: it uses a species-specific base GRN (bundled in
    the Docker image) and fits regularized regression on the given cells. GRNScope
    handles per-cluster scoping upstream (one cluster's cells per invocation), so
    this runner treats the provided cells as a single GRN unit.
    """

    def generateInputs(self):
        '''
        Write the expression matrix (genes x cells) into working_dir. CellOracle
        reads it and builds an AnnData internally, so no reformatting is needed.
        '''
        CELLORACLE_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if not CELLORACLE_EXPRESSION_FILE.exists():
            ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header=0, index_col=0)
            ExpressionData.to_csv(CELLORACLE_EXPRESSION_FILE,
                                  sep=',', header=True, index=True)

    def run(self):
        '''
        Function to run CellOracle inside the Docker image.
        '''
        species = str(self.params.get('species', 'human'))
        base_grn = str(self.params.get('baseGrn', 'auto'))
        alpha = str(self.params.get('alpha', 10))
        p_value_cutoff = str(self.params.get('pValueCutoff', 0.05))
        top_k = str(self.params.get('topK', 0))
        max_genes = str(self.params.get('maxGenes', 0))
        max_cells = str(self.params.get('maxCells', 0))

        cmdToRun = ' '.join(['docker run --rm',
                            f"-v {self.working_dir}:/usr/working_dir",
                            f'{self.image} /bin/sh -c \"time -v -o',
                            "/usr/working_dir/time.txt",
                            'python runCellOracle.py',
                            '--inFile', "/usr/working_dir/ExpressionData.csv",
                            '--outFile', "/usr/working_dir/outFile.txt",
                            '--species', species,
                            '--baseGrn', base_grn,
                            '--alpha', alpha,
                            '--pValueCutoff', p_value_cutoff,
                            '--topK', top_k,
                            '--maxGenes', max_genes,
                            '--maxCells', max_cells, '\"'])

        self._run_docker(cmdToRun)

    def _resolve_top_k(self):
        '''
        Resolve the maximum number of edges to keep per target gene from
        maxRegulatorsPerTarget. Returns None when absent (standalone BEELINE).
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
        Function to parse outputs from CellOracle.
        '''
        workDir = self.working_dir
        outFile = workDir / 'outFile.txt'

        # Quit if output file does not exist
        if not outFile.exists():
            print(str(outFile) + ' does not exist, skipping...')
            return

        top_k = self._resolve_top_k()

        # CellOracle's output is already sparse (base-GRN-limited); the heap keeps
        # only the top-K edges per target (Gene2) by absolute weight to match
        # GRNScope's downstream per-target cap without loading everything.
        if top_k is not None:
            self._parse_output_topk(outFile, top_k)
            return

        self._parse_output_full(outFile)

    def _parse_output_topk(self, outFile, top_k):
        target_heaps: dict = {}
        sequence = 0
        with outFile.open('r', newline='') as handle:
            reader = csv.reader(handle, delimiter='\t')
            next(reader, None)  # header: Gene1 Gene2 EdgeWeight
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

    def _parse_output_full(self, outFile):
        OutDF = pd.read_csv(outFile, sep='\t', header=0)
        self._write_ranked_edges(OutDF[['Gene1', 'Gene2', 'EdgeWeight']])
