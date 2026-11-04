import csv
import os
from pathlib import Path
import shlex
import shutil
import pandas as pd

from BLRun.runner import Runner


class PPCORRunner(Runner):
    """Concrete runner for the PPCOR GRN inference algorithm."""

    DEFAULT_MAX_GENES = 500
    MINIMUM_GENES = 3

    def generateInputs(self):
        '''
        Make the expression matrix available to PPCOR inside working_dir.

        PPCOR needs fewer genes than cells to provide the p-values GRNScope uses
        for edge filtering. Apply its mandatory configured/cell-count limit
        before the R process starts. Small already-safe matrices are hard-linked
        into working_dir (falling back to a copy across filesystems).
        '''
        source_file = self.input_dir / self.exprData
        PPCOR_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if PPCOR_EXPRESSION_FILE.exists():
            return
        max_genes = self._resolve_max_genes()
        if (
            getattr(self, 'expression_source', None) is not None
            or getattr(self, 'selected_cells_file', None) is not None
        ):
            expression = self.read_expression_data()
            self._limit_expression_genes(expression, max_genes).to_csv(
                PPCOR_EXPRESSION_FILE
            )
            return
        if not source_file.exists():
            raise FileNotFoundError(
                f"Expression data file not found: {source_file}")

        gene_count, cell_count = self._expression_shape(source_file)
        effective_max_genes = self._effective_max_genes(
            configured_max_genes=max_genes,
            cell_count=cell_count,
        )
        if gene_count > effective_max_genes:
            expression = self.read_expression_data()
            self._limit_expression_genes(expression, max_genes).to_csv(
                PPCOR_EXPRESSION_FILE
            )
            return
        try:
            os.link(source_file, PPCOR_EXPRESSION_FILE)
        except OSError:
            shutil.copy2(source_file, PPCOR_EXPRESSION_FILE)

    def _resolve_max_genes(self):
        raw = self.params.get('maxGenes', self.DEFAULT_MAX_GENES)
        try:
            max_genes = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("PPCOR maxGenes must be an integer.") from exc
        if max_genes < self.MINIMUM_GENES:
            raise ValueError(
                f"PPCOR maxGenes must be at least {self.MINIMUM_GENES}."
            )
        return max_genes

    @staticmethod
    def _expression_shape(source_file):
        with source_file.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                raise ValueError("PPCOR expression data is empty.")
            cell_count = max(0, len(header) - 1)
            gene_count = sum(
                1
                for row in reader
                if row and any(str(value).strip() for value in row)
            )
        return gene_count, cell_count

    def _effective_max_genes(self, *, configured_max_genes, cell_count):
        cell_limited_max_genes = int(cell_count) - 1
        if cell_limited_max_genes < self.MINIMUM_GENES:
            raise ValueError(
                "PPCOR requires at least 4 cells so it can analyze at least "
                "3 genes with valid partial-correlation p-values."
            )
        return min(int(configured_max_genes), cell_limited_max_genes)

    def _limit_expression_genes(self, expression, configured_max_genes):
        effective_max_genes = self._effective_max_genes(
            configured_max_genes=configured_max_genes,
            cell_count=len(expression.columns),
        )
        if len(expression.index) <= effective_max_genes:
            return expression
        variances = expression.var(axis=1, ddof=0)
        retained_genes = (
            variances.sort_values(ascending=False, kind='mergesort')
            .head(effective_max_genes)
            .index
        )
        return expression.loc[retained_genes]

    def run(self):
        '''
        Function to run PPCOR algorithm
        '''

        script_path = (
            Path(__file__).resolve().parents[1]
            / 'Algorithms' / 'PPCOR' / 'runPPCOR.R'
        )
        top_k = self._resolve_top_k() or 0
        p_value_cutoff = str(self.params['pVal'])
        cmdToRun = ' '.join(['docker run --rm',
                            f"-v {shlex.quote(str(self.working_dir))}:/usr/working_dir",
                            f"-v {shlex.quote(str(script_path))}:/runPPCOR.R:ro",
                            f'{self.image} /bin/sh -c \"time -v -o',
                            "/usr/working_dir/time.txt",
                            'Rscript /runPPCOR.R',
                            "/usr/working_dir/ExpressionData.csv",
                            "/usr/working_dir/outFile.txt",
                            p_value_cutoff, str(top_k), '\"'])

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
            raise FileNotFoundError(outFile)

        p_val_cutoff = float(self.params['pVal'])
        top_k = self._resolve_top_k()

        header = pd.read_csv(outFile, sep='\t', nrows=0).columns
        if 'EdgeWeight' in header:
            compact = pd.read_csv(outFile, sep='\t', converters={0: str, 1: str})
            self._write_ranked_edges(
                compact[['Gene1', 'Gene2', 'EdgeWeight']]
            )
            return

        if top_k is None:
            # No per-target cap (e.g. standalone BEELINE): reproduce the original
            # full-output behaviour exactly.
            self._parse_output_full(outFile, p_val_cutoff)
            return

        def edges():
            with outFile.open('r', newline='') as handle:
                reader = csv.DictReader(handle, delimiter='\t')
                if not {'Gene1', 'Gene2', 'corVal', 'pValue'}.issubset(reader.fieldnames or []):
                    raise ValueError("PPCOR output is missing expected columns.")
                for row in reader:
                    weight = float(row['corVal']) if float(row['pValue']) <= p_val_cutoff else 0.0
                    yield row['Gene1'], row['Gene2'], weight
        self._write_ranked_edges(self._merge_bounded_trajectory_edges([edges()], top_k))

    def _parse_output_full(self, outFile, p_val_cutoff):
        '''
        Original full-output parse: keep every edge, significant edges (ranked by
        absolute partial correlation) first with their signed weight, then the
        remaining edges with weight 0.
        '''
        OutDF = pd.read_csv(outFile, sep='\t', header=0, converters={0: str, 1: str})
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
