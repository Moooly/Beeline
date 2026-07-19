import csv
import heapq
import os
from pathlib import Path
import shlex
import pandas as pd

from BLRun.runner import Runner


class PIDCRunner(Runner):
    """Concrete runner for the PIDC GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for PIDC.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        # Create ExpressionData.csv file in the created input directory
        PIDC_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if not PIDC_EXPRESSION_FILE.exists():
            ExpressionData = self.read_expression_data()
            max_genes = self._resolve_max_genes()
            if max_genes is not None and len(ExpressionData.index) > max_genes:
                # Stable variance ranking keeps ties in the source-matrix order.
                # GRNScope normally applies this cap once before confidence
                # subsampling; this runner-side guard keeps standalone BEELINE
                # use consistent with the exposed parameter.
                variances = ExpressionData.var(axis=1, ddof=0)
                retained_genes = (
                    variances.sort_values(ascending=False, kind='mergesort')
                    .head(max_genes)
                    .index
                )
                ExpressionData = ExpressionData.loc[retained_genes]
            ExpressionData.to_csv(PIDC_EXPRESSION_FILE,
                                 sep = '\t', header  = True, index = True)

    def _resolve_max_genes(self):
        raw = self.params.get('maxGenes')
        if raw is None:
            return None
        try:
            max_genes = int(raw)
        except (TypeError, ValueError):
            return None
        return max_genes if max_genes >= 3 else None

    def run(self):
        '''
        Function to run PIDC algorithm
        '''

        script_path = (
            Path(__file__).resolve().parents[1]
            / 'Algorithms' / 'PIDC' / 'runPIDC.jl'
        )
        top_k = str(self._resolve_top_k() or 0)
        cmdToRun = ' '.join(['docker run --rm',
                            f"-v {shlex.quote(str(self.working_dir))}:/usr/working_dir",
                            f"-v {shlex.quote(str(script_path))}:/runPIDC.jl:ro",
                            f'{self.image} /bin/sh -c \"time -v -o',
                            "/usr/working_dir/time.txt",
                            'julia /runPIDC.jl',
                            "/usr/working_dir/ExpressionData.csv",
                            "/usr/working_dir/outFile.txt", top_k, '\"'])

        self._run_docker(cmdToRun)

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
        Function to parse outputs from PIDC.
        '''
        workDir = self.working_dir
        outFile = workDir / 'outFile.txt'

        # Quit if output file does not exist
        if not outFile.exists():
            print(str(outFile) + ' does not exist, skipping...')
            return

        top_k = self._resolve_top_k()

        # PIDC has no pseudotime/trajectories, so there is a single output file
        # and no cross-trajectory merge. Keeping only the top-K edges per target
        # (by |EdgeWeight|) in a heap is exact and never loads the full g^2 edge
        # list into memory.
        if top_k is not None:
            self._parse_output_topk(outFile, top_k)
            return

        self._parse_output_full(outFile)

    def _parse_output_topk(self, outFile, top_k):
        '''
        Stream the headerless edge list and keep only the top-K edges per target
        (Gene2, column 1) by absolute weight in a heap, matching GRNScope's
        downstream per-target cap without loading the full g^2 list.
        '''
        target_heaps: dict = {}
        sequence = 0
        with outFile.open('r', newline='') as handle:
            reader = csv.reader(handle, delimiter='\t')
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    weight = float(row[2])
                except ValueError:
                    continue
                gene1 = row[0]
                gene2 = row[1]
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
        '''
        Original full parse: pass every edge through unchanged.
        '''
        # Read output (headerless: col 0 = Gene1, col 1 = Gene2, col 2 = EdgeWeight)
        OutDF = pd.read_csv(outFile, sep = '\t', header = None)

        self._write_ranked_edges(pd.DataFrame({
            'Gene1':      OutDF[0],
            'Gene2':      OutDF[1],
            'EdgeWeight': OutDF[2],
        }))
