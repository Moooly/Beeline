import numpy as np
import pandas as pd

from BLRun.runner import Runner


# When no per-target cap is supplied (e.g. BEELINE run standalone, outside
# GRNScope), every edge is kept — preserving the original full-matrix behaviour.
# GRNScope always passes 'maxRegulatorsPerTarget', which activates the
# memory-bounded top-K path below.
_BLOCK_TARGET_ELEMENTS = 4_000_000


class PearsonRunner(Runner):
    """Concrete runner for pairwise Pearson correlation GRN inference.
    Runs entirely within the BEELINE conda environment; no Docker image is used.
    The image field in the config should be set to 'local'.

    The correlation matrix is computed with a single BLAS matrix multiply
    (standardize each gene, then Z @ Z.T) instead of pandas' pairwise
    ``DataFrame.corr``. Only the strongest ``maxRegulatorsPerTarget`` edges per
    target are kept, and the computation is streamed in row-blocks, so peak
    memory grows as O(genes x K) rather than O(genes^2). The result is the same
    ranked edge list the previous implementation produced.
    """

    def generateInputs(self):
        '''
        Verifies that the expression data file exists in the input directory.
        No file copying is required because Pearson runs locally without Docker.

        :param self.input_dir: Path — directory containing input files
        :param self.exprData: str — expression data filename
        :raises FileNotFoundError: if the expression data file is missing
        '''
        if not (self.input_dir / self.exprData).exists():
            raise FileNotFoundError(
                f"Expression data file not found: {self.input_dir / self.exprData}")

    def _resolve_top_k(self, n_genes):
        '''
        Resolve the maximum number of regulators (edges) to keep per target gene.

        GRNScope only retains the strongest ``maxRegulatorsPerTarget`` edges per
        target downstream, so keeping more here is wasted work and memory. When
        the parameter is absent (standalone BEELINE), returns None to keep every
        edge and reproduce the original full-matrix behaviour.

        :param n_genes: int — number of genes in the expression matrix
        :returns: int | None — per-target edge cap, or None to keep all
        '''
        raw = self.params.get('maxRegulatorsPerTarget', self.params.get('topK'))
        if raw is None:
            return None
        try:
            top_k = int(raw)
        except (TypeError, ValueError):
            return None
        if top_k <= 0:
            return None
        # A cap at or above (n_genes - 1) keeps every non-self edge anyway.
        return min(top_k, max(1, n_genes - 1))

    def _resolve_block_size(self, n_genes):
        '''
        Choose how many target genes to correlate per block.

        The transient block is (block_size x n_genes); sizing it to a fixed
        element budget keeps peak memory flat regardless of gene count.

        :param n_genes: int — number of genes in the expression matrix
        :returns: int — number of target rows to process per block (>= 1)
        '''
        if n_genes <= 0:
            return 1
        block_size = max(1, _BLOCK_TARGET_ELEMENTS // n_genes)
        return min(block_size, n_genes)

    def run(self):
        '''
        Computes pairwise Pearson correlation between all gene pairs and writes a
        ranked edge list (already reduced to the top edges per target) to
        working_dir/outFile.txt.

        Pearson correlation is scale-invariant, so the previous per-gene
        max-normalization is omitted: it never affected the output. Genes with
        zero variance yield undefined correlations (NaN), matching the behaviour
        of pandas' ``DataFrame.corr``; those edges are dropped.

        :param self.input_dir: Path — directory containing expression data
        :param self.exprData: str — CSV filename; rows = genes, columns = cells
        :param self.working_dir: Path — output location for outFile.txt
        :output working_dir/outFile.txt: tab-separated edge list with columns
            Gene1 (source), Gene2 (target), EdgeWeight (signed Pearson r)
        '''
        # Read expression data: rows = genes, columns = cells
        ExpressionData = pd.read_csv(
            self.input_dir / self.exprData, header=0, index_col=0)
        if not isinstance(ExpressionData, pd.DataFrame):
            raise TypeError(f"ExpressionData must be a DataFrame, got {type(ExpressionData)}")

        gene_names = [str(gene) for gene in ExpressionData.index]
        n_genes = len(gene_names)

        # Standardize each gene (row): center, then divide by its L2 norm. The
        # Pearson correlation matrix is then simply Z @ Z.T. All steps are done
        # in place so only a single genes x cells array is held (the operand of
        # every block multiply), instead of separate copies for the DataFrame,
        # centered, and normalized matrices.
        # .values.astype(float) (not .to_numpy) for compatibility with the older
        # pandas in the BEELINE environment; astype copies, so in-place ops below
        # are safe. genes x cells.
        standardized = ExpressionData.values.astype(float)
        del ExpressionData
        standardized -= standardized.mean(axis=1, keepdims=True)  # center in place
        norms = np.sqrt(np.einsum('ij,ij->i', standardized, standardized))
        # Zero-variance genes have undefined correlation. Divide safely, then
        # mark those rows NaN so every correlation involving them is NaN (dropped
        # downstream) — identical to pandas .corr() returning NaN for them.
        zero_variance = norms == 0
        norms[zero_variance] = 1.0
        standardized /= norms[:, None]  # normalize in place
        standardized[zero_variance, :] = np.nan

        top_k = self._resolve_top_k(n_genes)
        block_size = self._resolve_block_size(n_genes)

        with open(self.working_dir / 'outFile.txt', 'w') as out_file:
            out_file.write('Gene1\tGene2\tEdgeWeight\n')
            for block_start in range(0, n_genes, block_size):
                block_stop = min(block_start + block_size, n_genes)
                # block[a, s] = Pearson r between target (block_start + a) and source s
                block = standardized[block_start:block_stop] @ standardized.T
                for offset in range(block_stop - block_start):
                    target_index = block_start + offset
                    row = block[offset]
                    row[target_index] = np.nan  # exclude self-correlation
                    self._write_top_edges(
                        out_file, row, target_index, gene_names, top_k)

    def _write_top_edges(self, out_file, row, target_index, gene_names, top_k):
        '''
        Write the strongest edges pointing at a single target gene.

        Selects the top-K sources by absolute correlation using a partial sort
        (np.argpartition), so the full row is never fully sorted when only K
        edges are kept. Non-finite correlations (zero-variance genes) are
        excluded.

        :param out_file: open text file handle to append edge rows to
        :param row: np.ndarray — correlations of every source against this target
        :param target_index: int — index of the target gene (Gene2)
        :param gene_names: list[str] — gene labels indexed by row position
        :param top_k: int | None — max edges to keep for this target, or None
        '''
        finite_indices = np.flatnonzero(np.isfinite(row))
        if finite_indices.size == 0:
            return

        if top_k is not None and finite_indices.size > top_k:
            abs_values = np.abs(row[finite_indices])
            partitioned = np.argpartition(abs_values, finite_indices.size - top_k)
            selected = finite_indices[partitioned[finite_indices.size - top_k:]]
        else:
            selected = finite_indices

        # Rank the kept edges by absolute correlation, descending.
        selected = selected[np.argsort(np.abs(row[selected]))[::-1]]

        target_name = gene_names[target_index]
        for source_index in selected:
            out_file.write(
                f"{gene_names[source_index]}\t{target_name}\t{float(row[source_index])!r}\n")

    def parseOutput(self):
        '''
        Reads the ranked edge list from working_dir/outFile.txt and writes the
        final ranked edge list to output_dir/rankedEdges.csv.

        The edge list is already reduced to the top edges per target and excludes
        self-correlations; here it is sorted globally by absolute correlation,
        descending, retaining the signed EdgeWeight.

        :param self.working_dir: Path — directory containing outFile.txt
        :output output_dir/rankedEdges.csv: tab-separated edge list with columns
            Gene1 (str), Gene2 (str), EdgeWeight (float, signed Pearson r)
        '''
        outFile = self.working_dir / 'outFile.txt'
        if not outFile.exists():
            print(str(outFile) + ' does not exist, skipping...')
            return

        OutDF = pd.read_csv(outFile, sep='\t', header=0)
        if not isinstance(OutDF, pd.DataFrame):
            raise TypeError(f"OutDF must be a DataFrame, got {type(OutDF)}")

        if OutDF.empty:
            self._write_ranked_edges(OutDF.reindex(columns=['Gene1', 'Gene2', 'EdgeWeight']))
            return

        # Rank by absolute correlation value, descending; retain signed EdgeWeight
        OutDF = OutDF.iloc[OutDF['EdgeWeight'].abs().argsort()[::-1]]

        self._write_ranked_edges(OutDF[['Gene1', 'Gene2', 'EdgeWeight']])
