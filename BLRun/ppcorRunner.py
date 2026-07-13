import csv
import heapq
import os
import shutil
import pandas as pd

from BLRun.runner import Runner


class PPCORRunner(Runner):
    """Concrete runner for the PPCOR GRN inference algorithm."""

    def generateInputs(self):
        '''
        Make the expression matrix available to PPCOR inside working_dir.

        PPCOR consumes the matrix in exactly the comma-separated genes x cells
        layout it is provided in, so no reformatting is needed. The file is
        hard-linked into working_dir (falling back to a copy across
        filesystems) instead of being read into pandas and rewritten verbatim,
        which avoids a redundant full read+write of the matrix on every run.
        '''
        source_file = self.input_dir / self.exprData
        PPCOR_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if PPCOR_EXPRESSION_FILE.exists():
            return
        if not source_file.exists():
            raise FileNotFoundError(
                f"Expression data file not found: {source_file}")
        try:
            os.link(source_file, PPCOR_EXPRESSION_FILE)
        except OSError:
            shutil.copy2(source_file, PPCOR_EXPRESSION_FILE)

    def run(self):
        '''
        Function to run PPCOR algorithm
        '''

        cmdToRun = ' '.join(['docker run --rm',
                            f"-v {self.working_dir}:/usr/working_dir",
                            f'{self.image} /bin/sh -c \"time -v -o',
                            "/usr/working_dir/time.txt",
                            'Rscript runPPCOR.R',
                            "/usr/working_dir/ExpressionData.csv", "/usr/working_dir/outFile.txt", '\"'])

        # Run command
        self._run_docker(cmdToRun)

    def _resolve_top_k(self):
        '''
        Resolve the maximum number of edges to keep per target gene.

        GRNScope keeps only the strongest ``maxRegulatorsPerTarget`` edges per
        target downstream, so retaining more here just materialises the full
        g x g edge list for nothing. Returns None when the cap is absent
        (standalone BEELINE), which preserves the original full output.
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
        Function to parse outputs from PPCOR.
        '''
        workDir = self.working_dir
        outFile = workDir / 'outFile.txt'

        # Quit if output file does not exist
        if not outFile.exists():
            print(str(outFile) + ' does not exist, skipping...')
            return

        p_val_cutoff = float(self.params['pVal'])
        top_k = self._resolve_top_k()

        if top_k is None:
            # No per-target cap (e.g. standalone BEELINE): reproduce the original
            # full-output behaviour exactly.
            self._parse_output_full(outFile, p_val_cutoff)
            return

        # Bounded path: stream PPCOR's g x g edge list and keep only the top-K
        # edges per target in a heap, so the full g^2 list is never loaded into
        # memory. The edge weight matches the original: the signed partial
        # correlation for significant edges (p <= cutoff), else 0.0. Self-loops
        # and ranking are left as before so GRNScope's downstream top-K over this
        # output is unchanged.
        target_heaps: dict = {}
        sequence = 0
        with outFile.open('r', newline='') as handle:
            reader = csv.reader(handle, delimiter='\t')
            header = next(reader, None)
            if header is None:
                self._write_ranked_edges(
                    pd.DataFrame(columns=['Gene1', 'Gene2', 'EdgeWeight'])
                )
                return
            column = {name: index for index, name in enumerate(header)}
            try:
                gene1_col, gene2_col = column['Gene1'], column['Gene2']
                cor_col, pval_col = column['corVal'], column['pValue']
            except KeyError as exc:
                raise ValueError(f"PPCOR output missing expected column: {exc}")

            widest_column = max(gene1_col, gene2_col, cor_col, pval_col)
            for row in reader:
                if len(row) <= widest_column:
                    continue
                try:
                    cor_val = float(row[cor_col])
                    p_value = float(row[pval_col])
                except ValueError:
                    continue
                gene1, gene2 = row[gene1_col], row[gene2_col]
                weight = cor_val if p_value <= p_val_cutoff else 0.0
                abs_weight = abs(weight)
                heap = target_heaps.setdefault(gene2, [])
                item = (abs_weight, sequence, gene1, gene2, weight)
                sequence += 1
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif abs_weight > heap[0][0]:
                    heapq.heapreplace(heap, item)

        ranked_rows = []
        for heap in target_heaps.values():
            for abs_weight, _seq, gene1, gene2, weight in sorted(
                heap, key=lambda entry: (-entry[0], entry[1])
            ):
                ranked_rows.append((gene1, gene2, weight))

        self._write_ranked_edges(
            pd.DataFrame(ranked_rows, columns=['Gene1', 'Gene2', 'EdgeWeight'])
        )

    def _parse_output_full(self, outFile, p_val_cutoff):
        '''
        Original full-output parse: keep every edge, significant edges (ranked by
        absolute partial correlation) first with their signed weight, then the
        remaining edges with weight 0.
        '''
        OutDF = pd.read_csv(outFile, sep='\t', header=0)
        # edges with significant p-value
        part1 = OutDF.loc[OutDF['pValue'] <= p_val_cutoff]
        part1 = part1.assign(absCorVal=part1['corVal'].abs())
        # edges without significant p-value
        part2 = OutDF.loc[OutDF['pValue'] > p_val_cutoff]

        part1_sorted = part1.sort_values('absCorVal', ascending=False)
        part2_out = part2[['Gene1', 'Gene2']].copy()
        part2_out['EdgeWeight'] = 0.0

        self._write_ranked_edges(pd.concat([
            part1_sorted[['Gene1', 'Gene2']].assign(EdgeWeight=part1_sorted['corVal']),
            part2_out,
        ], ignore_index=True)[['Gene1', 'Gene2', 'EdgeWeight']])
