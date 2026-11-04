import csv
from pathlib import Path
import shlex
import pandas as pd

from BLRun.runner import Runner


class SINCERITIESRunner(Runner):
    """Concrete runner for the SINCERITIES GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for SINCERITIES.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        ExpressionData = self.read_expression_data()
        PTData = self.read_pseudotime_data()
        max_genes = self._resolve_max_genes()
        if max_genes is not None:
            # The sign step computes a partial-correlation matrix across all
            # genes separately for each trajectory. Keep fewer genes than the
            # smallest trajectory's cell count so every matrix is estimable.
            records = self._selected_cell_records()
            source_by_output = ({record['output']: record['source'] for record in records}
                                if records is not None else {})
            trajectory_cell_counts = [
                len({source_by_output.get(str(cell), str(cell))
                     for cell in PTData.index[PTData[column].notna()]})
                for column in PTData.columns
            ]
            available_cells = min(
                trajectory_cell_counts or [len(ExpressionData.columns)]
            )
            if available_cells < 3:
                raise ValueError("SINCERITIES needs at least 3 distinct source cells in every trajectory.")
            max_genes = min(max_genes, available_cells - 1)
            if len(ExpressionData.index) > max_genes:
                variances = ExpressionData.var(axis=1, ddof=0)
                retained_genes = (
                    variances.sort_values(ascending=False, kind='mergesort')
                    .head(max_genes)
                    .index
                )
                ExpressionData = ExpressionData.loc[retained_genes]

        n_bins = int(self.params['nBins'])
        if n_bins < 5:
            raise ValueError("SINCERITIES requires at least 5 time bins.")
        colNames = PTData.columns
        for idx in range(len(colNames)):
            # Select cells belonging to each pseudotime trajectory
            colName = colNames[idx]
            index = PTData[colName].index[PTData[colName].notnull()]
            exprName = "ExpressionData"+str(idx)+".csv"
            newExpressionData = ExpressionData.loc[:,index].T
            # Perform quantile binning as recommeded in the paper
            # http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html#pandas.qcut
            nBins = n_bins
            tQuantiles = pd.qcut(PTData.loc[index,colName], q = nBins, duplicates ='drop')
            if tQuantiles.isna().any() or tQuantiles.nunique() < 5:
                raise ValueError("SINCERITIES requires at least 5 distinct, nonempty time bins after quantile binning.")
            mid = [(a.left + a.right)/2 for a in tQuantiles]

            newExpressionData['Time'] = mid
            newExpressionData.to_csv(self.working_dir / exprName,
                                 sep = ',', header  = True, index = False)

    def _resolve_max_genes(self):
        raw = self.params.get('maxGenes')
        if raw is None:
            return None
        try:
            max_genes = int(raw)
        except (TypeError, ValueError):
            return None
        return max_genes if max_genes >= 2 else None

    def run(self):
        '''
        Function to run SINCERITIES algorithm
        '''

        PTData = self.read_pseudotime_data()
        top_k = str(self._resolve_top_k() or 0)
        script_path = (
            Path(__file__).resolve().parents[1]
            / 'Algorithms' / 'SINCERITIES' / 'runSinceritiesGRNScope.R'
        )

        colNames = PTData.columns
        commands = []
        for idx in range(len(colNames)):
            cmdToRun = ' '.join(['docker run --rm',
                                f"-v {shlex.quote(str(self.working_dir))}:/usr/working_dir",
                                f"-v {shlex.quote(str(script_path))}:/SINCERITIES/runSinceritiesGRNScope.R:ro",
                                f'{self.image} /bin/sh -c \"time -v -o',
                                "/usr/working_dir/time" + str(idx) + ".txt",
                                'Rscript runSinceritiesGRNScope.R',
                                "/usr/working_dir/ExpressionData" + str(idx) + ".csv",
                                "/usr/working_dir/outFile" + str(idx) + ".txt",
                                top_k, '\"'])

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
        Function to parse outputs from SINCERITIES.
        '''
        workDir = self.working_dir

        PTData = self.read_pseudotime_data()
        colNames = PTData.columns

        # A missing trajectory must fail the run instead of publishing partial results.
        for idx in range(len(colNames)):
            if not (workDir / ('outFile'+str(idx)+'.txt')).exists():
                raise FileNotFoundError(workDir / ('outFile'+str(idx)+'.txt'))

        top_k = self._resolve_top_k()

        if top_k is not None:
            merged = self._merge_bounded_trajectory_edges(
                (
                    self._iter_output_edges(
                        workDir / ('outFile' + str(idx) + '.txt')
                    )
                    for idx in range(len(colNames))
                ),
                top_k,
            )
            self._write_ranked_edges(merged)
            return

        # General path: original full parse across all trajectories.
        self._parse_output_full(workDir, colNames)

    @staticmethod
    def _iter_output_edges(out_path):
        with out_path.open('r', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    interaction = float(row['Interaction'])
                except (KeyError, TypeError, ValueError):
                    continue
                regulation_mode = str(row.get('Edges', '')).strip().casefold()
                if regulation_mode == 'activation':
                    signed_interaction = abs(interaction)
                elif regulation_mode == 'repression':
                    signed_interaction = -abs(interaction)
                elif regulation_mode == 'no regulation':
                    signed_interaction = 0.0
                else:
                    raise ValueError(
                        "SINCERITIES output contains an unsupported regulatory "
                        f"mode: {row.get('Edges')!r}."
                    )
                # Preserve the original SINCERITIES orientation correction.
                yield (
                    row.get('TargetGENES', ''),
                    row.get('SourceGENES', ''),
                    signed_interaction,
                )

    def _parse_output_topk(self, out_path, top_k):
        '''
        Stream a single trajectory's edge list, keeping only the top-K edges per
        target in a heap. SINCERITIES' output orientation is relabelled so that
        GRNScope's target (Gene2) is SINCERITIES' SourceGENES, matching the
        column swap in the original full parse. Ranking uses magnitude while the
        emitted edge retains its activation/repression sign.
        '''
        self._write_ranked_edges(
            self._merge_bounded_trajectory_edges(
                (self._iter_output_edges(out_path),),
                top_k,
            )
        )

    def _parse_output_full(self, workDir, colNames):
        '''
        Keep the strongest trajectory magnitude for each edge while preserving
        the activation/repression mode reported by SINCERITIES.
        '''
        merged = self._merge_trajectory_edges(
            self._iter_output_edges(
                workDir / ('outFile' + str(index) + '.txt')
            )
            for index in range(len(colNames))
        )
        self._write_ranked_edges(merged)
