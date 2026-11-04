from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import csv
import os
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd

from BLRun.edges import reduce_edges, reduce_frame


_DATAFRAME_CACHE = {}
_DATAFRAME_CACHE_LIMIT = 4
_DATAFRAME_CACHE_BYTES = 256 * 1024 * 1024
_DATAFRAME_CACHE_METRICS = {'hits': 0, 'misses': 0}


def input_cache_metrics():
    return {
        **_DATAFRAME_CACHE_METRICS,
        'entries': len(_DATAFRAME_CACHE),
    }


def _read_cached_dataframe(path: Path) -> pd.DataFrame:
    """Read a genes/cells CSV once per persistent BLRunner process."""
    resolved_path = path.resolve()
    file_stat = resolved_path.stat()
    cache_key = (
        str(resolved_path),
        file_stat.st_size,
        file_stat.st_mtime_ns,
    )
    cached = _DATAFRAME_CACHE.get(cache_key)
    if cached is not None:
        _DATAFRAME_CACHE_METRICS['hits'] += 1
        # Refresh insertion order for byte-bounded LRU eviction.
        _DATAFRAME_CACHE.pop(cache_key)
        _DATAFRAME_CACHE[cache_key] = cached
        return cached

    _DATAFRAME_CACHE_METRICS['misses'] += 1

    stale_keys = [key for key in _DATAFRAME_CACHE if key[0] == str(resolved_path)]
    for stale_key in stale_keys:
        _DATAFRAME_CACHE.pop(stale_key, None)
    with resolved_path.open(newline='', encoding='utf-8-sig') as handle:
        header = next(csv.reader(handle), [])
    if not header or any(not name.strip() for name in header[1:]):
        raise ValueError('Input must have nonempty column identifiers.')
    if len(set(header[1:])) != len(header[1:]):
        raise ValueError('Input column identifiers must be unique.')
    # Do not use index_col here: pandas may infer an index's numeric type
    # even when a converter was supplied. Convert the first column to an
    # index only after reading it as literal text (including names like NA).
    dataframe = pd.read_csv(resolved_path, header=0, converters={0: str})
    identifiers = dataframe.iloc[:, 0]
    dataframe = dataframe.iloc[:, 1:]
    dataframe.index = pd.Index(identifiers.array, name=header[0] or None)
    dataframe.columns = header[1:]
    if any(not value.strip() for value in dataframe.index):
        raise ValueError('Input row identifiers must be nonempty.')
    size = int(dataframe.memory_usage(index=True, deep=True).sum())
    if size <= _DATAFRAME_CACHE_BYTES:
        def cache_bytes():
            return sum(int(frame.memory_usage(index=True, deep=True).sum())
                       for frame in _DATAFRAME_CACHE.values())
        while _DATAFRAME_CACHE and (len(_DATAFRAME_CACHE) >= _DATAFRAME_CACHE_LIMIT
                                   or cache_bytes() + size > _DATAFRAME_CACHE_BYTES):
            _DATAFRAME_CACHE.pop(next(iter(_DATAFRAME_CACHE)))
        _DATAFRAME_CACHE[cache_key] = dataframe
    return dataframe

class Runner(ABC):
    """
    Abstract base_input class for BEELINE GRN inference algorithm runners.

    Subclasses must implement generateInputs, run, and parseOutput.
    Attributes set here reflect the fields accessed by runner implementations.
    """

    def release_input_views(self):
        """Release batch-local copies once output parsing has finished."""
        self._expression_view_cache = None
        self._pseudotime_view_cache = None
        self._selected_cells_cache = None

    def __init__(self, root: Path, config: dict):
        """
        Parameters
        ----------
        root : Path
            Root path from which all subpaths are resolved.
        config : dict
            Merged configuration for a single dataset + algorithm run.
            Expected structure:
              input:
                input_dir:   <str>  input directory (absolute, or relative to root)
              dataset:
                dataset_id:          <str>  dataset group label (path segment under input_dir)
                run_id:              <str>  run label (path segment under dataset_id)
                exprData:            <str>  expression data filename
                pseudoTimeData:      <str>  pseudotime data filename
                groundTruthNetwork:  <str>  ground truth network filename
              output_settings:
                output_dir: <str>  output directory (absolute, or relative to root)
              algo_name: <str>  name of the algorithm (appended to output_dir)
              params: <dict>  algorithm-specific parameters
        """
        for label, value in [
            ('dataset_id', config['dataset']['dataset_id']),
            ('run_id', config['dataset']['run_id']),
            ('algo_name', config['algo_name']),
            ('experiment_id', config['output_settings'].get('experiment_id', '')),
        ]:
            if label == 'experiment_id' and value == '':
                continue
            if not isinstance(value, str) or not value or value in ('.', '..') or any(c in value for c in '/\\\0'):
                raise ValueError(label + ' must be a single nonempty path segment.')
        inp = config['input']
        ds  = config['dataset']

        input_dir_path  = Path(inp['input_dir'])
        output_dir_path = Path(config['output_settings']['output_dir'])
        # experiment_id : str — optional; when set, an experiment_id segment is
        # inserted between output_dir and the dataset path so multiple experiment
        # runs can coexist under the same base output directory.
        experiment_id_prefix = config['output_settings'].get('experiment_id', '')

        base_input = input_dir_path if input_dir_path.is_absolute() else root / input_dir_path

        base_output = output_dir_path if output_dir_path.is_absolute() else root / output_dir_path
        if experiment_id_prefix:
            base_output = base_output / experiment_id_prefix
        base_output = base_output / ds['dataset_id'] / ds['run_id'] / config['algo_name']

        base_input = base_input.resolve()
        base_output = base_output.resolve()

        # input_dir: run-level input directory (expression data, pseudo-time).
        self.input_dir  = base_input / ds['dataset_id'] / ds['run_id']
        self.output_dir = base_output
        self.working_dir = base_output / "working_dir"

        # Precompute progress message for CLI output.
        self.running_message = (
            f"Running {config['algo_name']} | dataset: {ds['dataset_id']} | run: {ds['run_id']}"
        )

        self.exprData           = ds.get('exprData',           'ExpressionData.csv')
        self.pseudoTimeData     = ds.get('pseudoTimeData',     'PseudoTime.csv')
        expression_source = ds.get('expressionSource')
        pseudotime_source = ds.get('pseudoTimeSource')
        selected_cells_file = ds.get('selectedCellsFile')
        self.expression_source = (
            Path(str(expression_source)).resolve() if expression_source else None
        )
        self.pseudotime_source = (
            Path(str(pseudotime_source)).resolve() if pseudotime_source else None
        )
        self.selected_cells_file = (
            Path(str(selected_cells_file)).resolve() if selected_cells_file else None
        )
        self._selected_cells_cache = None
        self._selected_cell_records_cache = None
        self._expression_view_cache = None
        self._pseudotime_view_cache = None
        self.groundTruthNetwork = ds.get('groundTruthNetwork', 'GroundTruthNetwork.csv')
        # ground_truth_file: full path to the dataset-level ground truth CSV.
        self.ground_truth_file  = base_input / ds['dataset_id'] / self.groundTruthNetwork

        # image: Docker image name used to run this algorithm (e.g. "grnbeeline/genie3:base").
        # Mandatory — every algorithm entry in the config must supply this field.
        if 'image' not in config or not config['image']:
            raise ValueError("Algorithm config must include a non-empty 'image' field.")
        if not isinstance(config['image'], str):
            raise TypeError(f"'image' must be a str, got {type(config['image'])}")
        self.image = config['image']

        # Unwrap single-element lists so runners receive scalar values.
        # YAML config files commonly wrap param values in brackets
        # (e.g. `pVal: [0.01]`), which YAML parses as a list.
        raw_params = config.get('params', {})
        self.params = {
            k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
            for k, v in raw_params.items()
        }
        self.resource_settings = dict(config.get('resources') or {})
        self.cpu_budget = max(
            1,
            int(self.resource_settings.get('cpu_budget', 1)),
        )
        raw_memory_budget = self.resource_settings.get('memory_budget_mb')
        self.memory_budget_mb = (
            max(512, int(raw_memory_budget))
            if raw_memory_budget is not None
            else None
        )
        self.trajectory_workers = max(
            1,
            int(self.resource_settings.get('trajectory_workers', self.cpu_budget)),
        )

        # Refuse cleanup if a misconfigured output would contain source data.
        for source in (self.expression_input_path(), self.pseudotime_input_path()):
            if self.working_dir.resolve() in source.resolve().parents:
                raise ValueError('Working directory must not contain input files.')
        if self.working_dir.is_symlink():
            raise ValueError('Working directory must not be a symbolic link.')
        # Erase working directory so stale inputs from prior runs are not reused.
        if self.working_dir.exists():
            for item in sorted(self.working_dir.rglob('*'), reverse=True):
                item.unlink() if (item.is_file() or item.is_symlink()) else item.rmdir()
            self.working_dir.rmdir()

        # Pre-create output_dir and working_dir so docker cannot claim them as root.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir.mkdir(parents=True, exist_ok=True)


    @contextmanager
    def resource_environment(self):
        thread_variables = {
            'OMP_NUM_THREADS': str(self.cpu_budget),
            'OPENBLAS_NUM_THREADS': str(self.cpu_budget),
            'MKL_NUM_THREADS': str(self.cpu_budget),
            'NUMEXPR_NUM_THREADS': str(self.cpu_budget),
            'VECLIB_MAXIMUM_THREADS': str(self.cpu_budget),
        }
        previous_values = {
            key: os.environ.get(key) for key in thread_variables
        }
        os.environ.update(thread_variables)
        try:
            try:
                from threadpoolctl import threadpool_limits
            except ImportError:
                yield
            else:
                with threadpool_limits(limits=self.cpu_budget):
                    yield
        finally:
            for key, previous_value in previous_values.items():
                if previous_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous_value

    def _selected_cell_records(self):
        selected_cells_file = getattr(self, 'selected_cells_file', None)
        if selected_cells_file is None:
            return None
        if getattr(self, '_selected_cell_records_cache', None) is None:
            values = json.loads(
                selected_cells_file.read_text(encoding='utf-8')
            )
            if not isinstance(values, list):
                raise ValueError('selectedCellsFile must contain a JSON list.')
            records = []
            for value in values:
                if isinstance(value, dict):
                    source = str(value.get('source', '')).strip()
                    output = str(value.get('output', '')).strip()
                else:
                    source = str(value).strip()
                    output = source
                if not source or not output:
                    raise ValueError(
                        'selectedCellsFile entries require non-empty source and output names.'
                    )
                records.append({'source': source, 'output': output})
            output_names = [record['output'] for record in records]
            if len(output_names) != len(set(output_names)):
                raise ValueError(
                    'selectedCellsFile output cell identifiers must be unique.'
                )
            self._selected_cell_records_cache = records
            self._selected_cells_cache = [record['source'] for record in records]
        return self._selected_cell_records_cache

    def _selected_cell_names(self):
        records = self._selected_cell_records()
        if records is None:
            return None
        return [record['source'] for record in records]

    def expression_input_path(self) -> Path:
        return getattr(self, 'expression_source', None) or (
            self.input_dir / self.exprData
        )

    def pseudotime_input_path(self) -> Path:
        return getattr(self, 'pseudotime_source', None) or (
            self.input_dir / self.pseudoTimeData
        )

    def read_expression_data(self) -> pd.DataFrame:
        cached_view = getattr(self, '_expression_view_cache', None)
        if cached_view is not None:
            return cached_view.copy(deep=False)

        expression = _read_cached_dataframe(self.expression_input_path())
        if expression.index.has_duplicates or expression.columns.has_duplicates:
            raise ValueError('Expression gene and cell identifiers must be unique.')
        selected_records = self._selected_cell_records()
        if selected_records is None:
            selected_expression = expression.copy()
            self._expression_view_cache = selected_expression
            return selected_expression.copy(deep=False)

        column_lookup = {str(column): column for column in expression.columns}
        missing = [
            record['source']
            for record in selected_records
            if record['source'] not in column_lookup
        ]
        if missing:
            raise ValueError(
                f'{len(missing)} selected cells are missing from the expression matrix.'
            )
        selected_columns = [
            column_lookup[record['source']]
            for record in selected_records
        ]
        selected_expression = expression.loc[:, selected_columns].copy()
        selected_expression.columns = [
            record['output']
            for record in selected_records
        ]
        self._expression_view_cache = selected_expression
        return selected_expression.copy(deep=False)

    def read_pseudotime_data(self) -> pd.DataFrame:
        cached_view = getattr(self, '_pseudotime_view_cache', None)
        if cached_view is not None:
            return cached_view.copy(deep=False)

        pseudotime = _read_cached_dataframe(self.pseudotime_input_path()).copy()
        if pseudotime.shape[1] == 0:
            raise ValueError(
                "Pseudotime contains no trajectory columns. Convert a "
                "one-value-per-row file to a cell-indexed PseudoTime column."
            )

        pseudotime.index = pd.Index(
            [str(value).strip() for value in pseudotime.index]
        )
        if pseudotime.index.has_duplicates:
            raise ValueError("Pseudotime contains duplicate cell identifiers.")

        numeric_pseudotime = pseudotime.apply(pd.to_numeric, errors="coerce")
        invalid_values = pseudotime.notna() & numeric_pseudotime.isna()
        if np.asarray(invalid_values).any():
            raise ValueError(
                "Pseudotime contains a non-numeric value outside blank/NA cells."
            )
        finite_values = np.asarray(numeric_pseudotime, dtype=float)
        if np.isinf(finite_values).any():
            raise ValueError("Pseudotime contains a non-finite value.")

        selected_records = self._selected_cell_records()
        expected_cells = [str(value) for value in self.read_expression_data().columns]
        row_lookup = {str(index): index for index in numeric_pseudotime.index}
        source_cells = (
            [record['source'] for record in selected_records]
            if selected_records is not None
            else expected_cells
        )
        missing_cells = [cell for cell in source_cells if cell not in row_lookup]
        if missing_cells:
            raise ValueError(
                f"Pseudotime is missing {len(missing_cells)} analyzed cells: "
                f"{', '.join(missing_cells[:5])}."
            )
        if selected_records is None:
            expected_set = set(expected_cells)
            unknown_cells = [
                str(index)
                for index in numeric_pseudotime.index
                if str(index) not in expected_set
            ]
            if unknown_cells:
                raise ValueError(
                    f"Pseudotime contains {len(unknown_cells)} unknown cells: "
                    f"{', '.join(unknown_cells[:5])}."
                )
        selected_rows = [row_lookup[cell] for cell in source_cells]
        selected_pseudotime = numeric_pseudotime.loc[selected_rows].copy()
        selected_pseudotime.index = pd.Index(expected_cells)
        selected_pseudotime = selected_pseudotime.dropna(
            axis=1,
            how="all",
        )
        if selected_pseudotime.shape[1] == 0:
            raise ValueError(
                "Pseudotime has no numeric trajectory values for the analyzed cells."
            )
        self._pseudotime_view_cache = selected_pseudotime
        return selected_pseudotime.copy(deep=False)

    @abstractmethod
    def generateInputs(self):
        """Prepare algorithm-specific input files from the dataset."""

    @abstractmethod
    def run(self):
        """Execute the inference algorithm."""

    @abstractmethod
    def parseOutput(self) -> None:
        """
        Parse raw algorithm output and write a ranked edge list to disk.

        Implementations should build a DataFrame with columns Gene1, Gene2,
        EdgeWeight and pass it to self._write_ranked_edges(). Raises an error if the expected raw output file is missing.
        """

    def _apply_docker_resource_limits(
        self,
        cmd: str,
        *,
        cpu_budget: int | None = None,
        memory_budget_mb: int | None = None,
    ) -> str:
        if not cmd.startswith('docker run '):
            return cmd

        resolved_cpu_budget = max(
            1,
            int(cpu_budget or getattr(self, 'cpu_budget', 1)),
        )
        resolved_memory_budget = (
            memory_budget_mb
            if memory_budget_mb is not None
            else getattr(self, 'memory_budget_mb', None)
        )
        flags = [
            f'--cpus={resolved_cpu_budget}',
            '-e', f'OMP_NUM_THREADS={resolved_cpu_budget}',
            '-e', f'OPENBLAS_NUM_THREADS={resolved_cpu_budget}',
            '-e', f'MKL_NUM_THREADS={resolved_cpu_budget}',
            '-e', f'NUMEXPR_NUM_THREADS={resolved_cpu_budget}',
            '-e', f'VECLIB_MAXIMUM_THREADS={resolved_cpu_budget}',
            '-e', f'NUMBA_NUM_THREADS={resolved_cpu_budget}',
            '-e', f'JULIA_NUM_THREADS={resolved_cpu_budget}',
        ]
        if resolved_memory_budget is not None:
            flags.append(f'--memory={max(512, int(resolved_memory_budget))}m')
        return cmd.replace('docker run ', f"docker run {' '.join(flags)} ", 1)

    def _run_docker(
        self,
        cmd: str,
        append: bool = False,
        *,
        cpu_budget: int | None = None,
        memory_budget_mb: int | None = None,
        output_path: Path | None = None,
    ) -> None:
        """
        Execute a shell command and write combined stdout/stderr to output.txt.

        Parameters
        ----------
        cmd : str
            Shell command to execute (passed to the shell verbatim).
        append : bool
            If True, append to an existing output.txt. Use for runners that
            invoke docker in a loop so all container output is collected in
            one file. Defaults to False (overwrite).
        """
        if not isinstance(cmd, str):
            raise TypeError(f"cmd must be str, got {type(cmd)}")
        if not isinstance(append, bool):
            raise TypeError(f"append must be bool, got {type(append)}")

        cmd = self._apply_docker_resource_limits(
            cmd,
            cpu_budget=cpu_budget,
            memory_budget_mb=memory_budget_mb,
        )
        resolved_output_path = output_path or (self.output_dir / 'output.txt')
        mode = 'a' if append else 'w'
        with open(resolved_output_path, mode) as f:
            proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                f.write(line)
                f.flush()
            proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"Docker command failed (exit {proc.returncode}). "
                f"See {resolved_output_path} for details."
            )

    def _trajectory_worker_count(self, command_count, max_workers=None):
        count = min(command_count, max(1, int(getattr(self, 'cpu_budget', 1))),
                    max(1, int(getattr(self, 'trajectory_workers', getattr(self, 'cpu_budget', 1)))))
        if max_workers is not None:
            count = min(count, max(1, int(max_workers)))
        memory = getattr(self, 'memory_budget_mb', None)
        if memory is not None:
            count = min(count, max(1, int(memory) // getattr(self, '_MIN_CONTAINER_MEMORY_MB', 512)))
        return max(1, count)

    def _run_docker_batch(
        self,
        commands,
        *,
        append: bool = False,
        max_workers: int | None = None,
        log_prefix: str = 'trajectory',
    ) -> None:
        commands = list(commands)
        if not commands:
            return
        if len(commands) == 1:
            self._run_docker(commands[0], append=append)
            return

        total_cpu_budget = max(1, int(getattr(self, 'cpu_budget', 1)))
        worker_count = self._trajectory_worker_count(len(commands), max_workers)
        self.execution_worker_count = max(worker_count, getattr(self, 'execution_worker_count', 1))
        if worker_count <= 1:
            for index, command in enumerate(commands):
                self._run_docker(command, append=append or index > 0)
            return

        cpu_per_worker = max(1, total_cpu_budget // worker_count)
        total_memory_budget = getattr(self, 'memory_budget_mb', None)
        memory_per_worker = (
            max(512, total_memory_budget // worker_count)
            if total_memory_budget is not None
            else None
        )
        log_paths = [
            self.output_dir / f'{log_prefix}-{index}.log'
            for index in range(len(commands))
        ]

        succeeded = False
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        self._run_docker,
                        command,
                        cpu_budget=cpu_per_worker,
                        memory_budget_mb=memory_per_worker,
                        output_path=log_paths[index],
                    )
                    for index, command in enumerate(commands)
                ]
                for future in futures:
                    future.result()
            succeeded = True
        finally:
            output_mode = 'a' if append else 'w'
            with (self.output_dir / 'output.txt').open(output_mode) as combined_output:
                for log_path in log_paths:
                    if log_path.is_file():
                        combined_output.write(log_path.read_text(errors='replace'))
                        if succeeded:
                            log_path.unlink()

    def _write_ranked_edges(self, df: pd.DataFrame) -> None:
        """
        Write a ranked edge list to self.output_dir/rankedEdges.csv.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns Gene1, Gene2, EdgeWeight.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If self.output_dir does not exist at the time of writing.
        TypeError
            If df is not a pd.DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be pd.DataFrame, got {type(df)}")
        if not self.output_dir.is_dir():
            raise FileNotFoundError(
                f"Output directory does not exist: {self.output_dir}")
        required_columns = ['Gene1', 'Gene2', 'EdgeWeight']
        missing_columns = [column for column in required_columns if column not in df]
        if missing_columns:
            raise ValueError(
                f"Ranked edge output is missing columns: {missing_columns}"
            )

        input_edge_count = int(len(df))
        top_k = self._resolve_output_top_k()
        bounded = self._top_edges_per_target(df[required_columns], top_k)
        temporary_path = self.output_dir / 'rankedEdges.csv.tmp'
        bounded.to_csv(
            temporary_path,
            sep='\t',
            index=False,
        )
        temporary_path.replace(self.output_dir / 'rankedEdges.csv')
        self.output_reduction_observability = {
            'input_edge_count': input_edge_count,
            'retained_edge_count': int(len(bounded)),
            'max_regulators_per_target': top_k,
            'ranked_edges_preserved': True,
        }

    def _resolve_output_top_k(self) -> int | None:
        raw = self.params.get('maxRegulatorsPerTarget')
        if raw is None:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @staticmethod
    def _top_edges_per_target(edges, top_k):
        return reduce_frame(edges, top_k)

    @classmethod
    def _merge_bounded_trajectory_edges(cls, edge_streams, top_k):
        return reduce_edges(edge_streams, top_k)

    @staticmethod
    def _merge_trajectory_edges(edge_streams):
        return reduce_edges(edge_streams)
