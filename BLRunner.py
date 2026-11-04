import argparse
import hashlib
import json
import os
import re
import sys
import subprocess
import platform
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
import yaml
from tqdm import tqdm

from BLRun.celloracleRunner import CellOracleRunner
from BLRun.celloracleRunner import shutdown_celloracle_workers
from BLRun.genie3Runner import GENIE3Runner
from BLRun.grnboost2Runner import GRNBoost2Runner
from BLRun.grisliRunner import GRISLIRunner
from BLRun.grnvbemRunner import GRNVBEMRunner
from BLRun.jump3Runner import JUMP3Runner
from BLRun.leapRunner import LEAPRunner
from BLRun.pidcRunner import PIDCRunner
from BLRun.ppcorRunner import PPCORRunner
from BLRun.scodeRunner import SCODERunner
from BLRun.scribeRunner import SCRIBERunner
from BLRun.scsglRunner import SCSGLRunner
from BLRun.sinceritiesRunner import SINCERITIESRunner
from BLRun.singeRunner import SINGERunner
from BLRun.pearsonRunner import PearsonRunner
from BLRun.runner import input_cache_metrics

RUNNERS = {
    'CELLORACLE':   CellOracleRunner,
    'GENIE3':       GENIE3Runner,
    'GRNBOOST2':    GRNBoost2Runner,
    'GRISLI':       GRISLIRunner,
    'GRNVBEM':      GRNVBEMRunner,
    'JUMP3':        JUMP3Runner,
    'LEAP':         LEAPRunner,
    'PEARSON':      PearsonRunner,
    'PIDC':         PIDCRunner,
    'PPCOR':        PPCORRunner,
    'SCODE':        SCODERunner,
    'SCRIBE':       SCRIBERunner,
    'SCSGL':        SCSGLRunner,
    'SINCERITIES':  SINCERITIESRunner,
    'SINGE':        SINGERunner,
}

PHASE_TIMINGS_FILENAME = 'phase_timings.json'
_FILE_DIGEST_CACHE = {}


def _rounded_seconds(value):
    return round(max(0.0, float(value)), 6)


def _sha256_file(path):
    try:
        is_file = path.is_file()
    except OSError:
        return None
    if not is_file:
        return None

    try:
        file_stat = path.stat()
    except OSError:
        return None
    cache_key = (str(path.resolve()), file_stat.st_size, file_stat.st_mtime_ns)
    cached = _FILE_DIGEST_CACHE.get(cache_key)
    if cached is not None:
        return cached

    digest = hashlib.sha256()
    try:
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
    except OSError:
        return None
    value = digest.hexdigest()
    _FILE_DIGEST_CACHE[cache_key] = value
    return value


def _parse_elapsed_wall_clock(raw_value):
    parts = raw_value.strip().split(':')
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None

    if len(values) == 2:
        minutes, seconds = values
        return minutes * 60 + seconds
    if len(values) == 3:
        hours, minutes, seconds = values
        return hours * 3600 + minutes * 60 + seconds
    return None


def collect_gnu_time_metrics(working_dir):
    """Aggregate GNU ``time -v`` output from one or more trajectories."""
    time_files = sorted(working_dir.glob('time*.txt'))
    if not time_files:
        return None

    total_user_seconds = 0.0
    total_system_seconds = 0.0
    total_wall_seconds = 0.0
    maximum_rss_kib = 0
    parsed_file_count = 0

    for time_file in time_files:
        text = time_file.read_text(encoding='utf-8', errors='replace')
        user_match = re.search(r'User time \(seconds\):\s+([\d.]+)', text)
        system_match = re.search(r'System time \(seconds\):\s+([\d.]+)', text)
        elapsed_match = re.search(
            r'Elapsed \(wall clock\) time.*?:\s*'
            r'([0-9]+(?::[0-9.]+){1,2})\s*$',
            text,
            flags=re.MULTILINE,
        )
        rss_match = re.search(
            r'Maximum resident set size \(kbytes\):\s+(\d+)',
            text,
        )

        elapsed_seconds = (
            _parse_elapsed_wall_clock(elapsed_match.group(1))
            if elapsed_match
            else None
        )
        if not (user_match and system_match and elapsed_seconds is not None):
            continue

        parsed_file_count += 1
        total_user_seconds += float(user_match.group(1))
        total_system_seconds += float(system_match.group(1))
        total_wall_seconds += elapsed_seconds
        if rss_match:
            maximum_rss_kib = max(maximum_rss_kib, int(rss_match.group(1)))

    if parsed_file_count == 0:
        return {
            'time_file_count': len(time_files),
            'parsed_time_file_count': 0,
        }

    cpu_seconds = total_user_seconds + total_system_seconds
    return {
        'time_file_count': len(time_files),
        'parsed_time_file_count': parsed_file_count,
        'container_wall_seconds': _rounded_seconds(total_wall_seconds),
        'cpu_user_seconds': _rounded_seconds(total_user_seconds),
        'cpu_system_seconds': _rounded_seconds(total_system_seconds),
        'cpu_total_seconds': _rounded_seconds(cpu_seconds),
        'effective_cpu_percent': round(
            (cpu_seconds / total_wall_seconds) * 100,
            2,
        ) if total_wall_seconds > 0 else None,
        'maximum_resident_set_kib': maximum_rss_kib or None,
        'wall_time_semantics': 'sum_of_container_elapsed_times',
        'rss_semantics': 'maximum_single_process_high_water_mark_not_aggregate',
    }


def _runner_fingerprints(runner):
    expression_path = (
        runner.expression_input_path()
        if hasattr(runner, 'expression_input_path')
        else runner.input_dir / runner.exprData
    )
    pseudotime_path = (
        runner.pseudotime_input_path()
        if hasattr(runner, 'pseudotime_input_path')
        else runner.input_dir / runner.pseudoTimeData
    )
    selected_cells_path = getattr(runner, 'selected_cells_file', None)
    ranked_edges_path = runner.output_dir / 'rankedEdges.csv'
    params_json = json.dumps(
        runner.params,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    ).encode('utf-8')
    return {
        'expression_sha256': _sha256_file(expression_path),
        'pseudotime_sha256': _sha256_file(pseudotime_path),
        'generated_inputs_sha256': {
            str(path.relative_to(runner.working_dir)): _sha256_file(path)
            for path in sorted(runner.working_dir.rglob('*'))
            if path.is_file() and (path.name.startswith(('ExpressionData', 'PseudoTime', 'pseudoTimeData', 'GeneList'))
                                   or path.name == 'hyperparameters.txt')
        },
        'selected_cells_sha256': (
            _sha256_file(selected_cells_path) if selected_cells_path else None
        ),
        'algorithm_parameters_sha256': hashlib.sha256(params_json).hexdigest(),
        'ranked_edges_sha256': _sha256_file(ranked_edges_path),
    }


def runtime_provenance(runner):
    """Identify code, image contents and host numerics, including uncommitted edits."""
    root = Path(__file__).resolve().parent
    files = [root / 'BLRunner.py'] + sorted((root / 'BLRun').glob('*.py'))
    files += sorted(path for path in (root / 'Algorithms').rglob('*')
                    if path.suffix in {'.py', '.R', '.jl', '.awk'} and path.is_file())
    signature = hashlib.sha256()
    for path in files:
        signature.update(str(path.relative_to(root)).encode())
        signature.update((_sha256_file(path) or '').encode())
    import numpy
    import pandas
    result = {
        'source_sha256': signature.hexdigest(),
        'python': platform.python_version(),
        'numpy': numpy.__version__, 'pandas': pandas.__version__,
        'parameters': dict(runner.params),
        'edge_policy': 'no_self_unique_max_absolute_positive_sign_tie_lexical_names_v1',
    }
    try:
        revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, timeout=5)
        result['source_revision'] = revision.stdout.strip() if revision.returncode == 0 else None
        if runner.image != 'local':
            inspection = subprocess.run(['docker', 'image', 'inspect', runner.image],
                                        capture_output=True, text=True, timeout=10)
            if inspection.returncode == 0:
                image = json.loads(inspection.stdout)[0]
                result['image_id'] = image['Id']
                result['image_digests'] = image.get('RepoDigests', [])
            else:
                result['image_inspection'] = 'unavailable'
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError):
        result['image_inspection'] = 'unavailable'
    return result


def execute_runner(runner):
    """Execute one runner and persist phase/resource observability metadata."""
    ranked_path = runner.output_dir / 'rankedEdges.csv'
    if ranked_path.exists():
        ranked_path.unlink()
    pipeline_started = time.perf_counter()
    cache_metrics_before = input_cache_metrics()
    payload = {
        'schema_version': 1,
        'algorithm_image': runner.image,
        'status': 'Running',
        'resource_allocation': getattr(runner, 'resource_settings', {}),
        'stages_seconds': {
            'runner_setup': _rounded_seconds(
                getattr(runner, '_phase_setup_seconds', 0.0)
            ),
        },
    }
    resource_context = (
        runner.resource_environment()
        if hasattr(runner, 'resource_environment')
        else nullcontext()
    )
    context_entered = False
    try:
        resource_context.__enter__()
        context_entered = True
        payload['provenance'] = runtime_provenance(runner)
        stage_started = time.perf_counter()
        runner.generateInputs()
        payload['stages_seconds']['input_generation'] = _rounded_seconds(
            time.perf_counter() - stage_started
        )

        stage_started = time.perf_counter()
        runner.run()
        execution_seconds = time.perf_counter() - stage_started
        payload['stages_seconds']['algorithm_execution'] = _rounded_seconds(
            execution_seconds
        )
        persistent_runtime = getattr(
            runner, 'persistent_runtime_observability', None
        )
        if persistent_runtime is not None:
            payload['persistent_runtime'] = persistent_runtime

        resource_usage = collect_gnu_time_metrics(runner.working_dir)
        if resource_usage is not None:
            if persistent_runtime is not None:
                resource_usage['rss_semantics'] = 'persistent_worker_lifetime_high_water_mark_not_per_request_peak'
            payload['resource_usage'] = resource_usage
            container_wall_seconds = resource_usage.get('container_wall_seconds')
            if container_wall_seconds is not None and getattr(runner, 'execution_worker_count', 1) == 1:
                payload['stages_seconds']['container_and_wrapper_overhead_estimate'] = (
                    _rounded_seconds(execution_seconds - container_wall_seconds)
                )

        stage_started = time.perf_counter()
        runner.parseOutput()
        if not ranked_path.is_file():
            raise FileNotFoundError(f'Algorithm did not produce fresh ranked edges: {ranked_path}')
        with ranked_path.open() as handle:
            if handle.readline().rstrip('\r\n') != 'Gene1\tGene2\tEdgeWeight':
                raise ValueError('Ranked edge output has an invalid header.')
        payload['stages_seconds']['output_parsing'] = _rounded_seconds(
            time.perf_counter() - stage_started
        )
        output_reduction = getattr(
            runner, 'output_reduction_observability', None
        )
        if output_reduction is not None:
            payload['output_reduction'] = output_reduction
        payload['status'] = 'Completed'
    except Exception as exc:
        if ranked_path.exists():
            ranked_path.unlink()
        payload['status'] = 'Failed'
        payload['error_type'] = type(exc).__name__
        raise
    finally:
        if hasattr(runner, 'release_input_views'):
            runner.release_input_views()
        if context_entered:
            resource_context.__exit__(*sys.exc_info())
        pipeline_seconds = time.perf_counter() - pipeline_started
        payload['stages_seconds']['runner_pipeline_total'] = _rounded_seconds(
            pipeline_seconds
        )
        payload['stages_seconds']['runner_observed_total'] = _rounded_seconds(
            pipeline_seconds + getattr(runner, '_phase_setup_seconds', 0.0)
        )
        if 'provenance' in payload:
            payload['provenance']['parameters'] = dict(runner.params)
        payload['fingerprints'] = _runner_fingerprints(runner)
        cache_metrics_after = input_cache_metrics()
        payload['input_cache'] = {
            'hits': cache_metrics_after['hits'] - cache_metrics_before['hits'],
            'misses': cache_metrics_after['misses'] - cache_metrics_before['misses'],
            'entries': cache_metrics_after['entries'],
        }
        runner.working_dir.mkdir(parents=True, exist_ok=True)
        (runner.working_dir / PHASE_TIMINGS_FILENAME).write_text(
            json.dumps(payload, indent=2),
            encoding='utf-8',
        )

    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description='BLRunner: Run GRN inference algorithms using BEELINE.'
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=False,
        help='Path to the configuration file used to run the inference algorithms.'
    )
    parser.add_argument(
        '--worker',
        action='store_true',
        help='Keep BLRunner alive and accept JSON-line run requests on stdin.',
    )
    parser.add_argument(
        '--ready-file',
        type=str,
        help='Optional path written atomically when worker initialization completes.',
    )
    parser.add_argument('--overwrite', action='store_true', help='Replace prior results for this configuration.')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def enabled(value, default=False):
    if value is None:
        return default
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, bool):
        raise ValueError("should_run must be a boolean or a one-element boolean list.")
    return value


def validate_path_segment(value, label, allow_empty=False):
    if allow_empty and value == '':
        return
    if not isinstance(value, str) or not value or value in ('.', '..') or any(c in value for c in '/\\\0'):
        raise ValueError(f"{label} must be a single nonempty path segment.")


def get_datasets(input_settings):
    """
    Return a flat list of dataset dicts from input_settings.

    Each dataset entry in the config becomes a separate runnable dataset, with
    each run forming a flat entry keyed by run_id. The dataset_id is used
    directly as the path segment beneath input_dir.

    If a dataset has 'scan_run_subdirectories: true', subdirectories of
    input_dir/dataset_id/ are discovered at runtime and used as runs instead
    of an explicit 'runs' list.

    If 'datasets' is absent, returns an empty list (caller handles auto-discovery).
    """
    if 'datasets' not in input_settings:
        return []

    datasets = []
    input_dir = Path.cwd() / input_settings['input_dir']

    for ds in input_settings['datasets']:
        if not enabled(ds.get('should_run'), True):
            continue

        validate_path_segment(ds['dataset_id'], 'dataset_id')
        if ds.get('scan_run_subdirectories'):
            # Discover runs by scanning subdirectories of the dataset input path.
            # ds_input_path : Path — input_dir/dataset_id/
            ds_input_path = input_dir / ds['dataset_id']
            if not ds_input_path.is_dir():
                raise FileNotFoundError(
                    f"scan_run_subdirectories is set for dataset '{ds['dataset_id']}' "
                    f"but input directory '{ds_input_path}' does not exist."
                )
            runs = [{'run_id': d.name} for d in sorted(ds_input_path.iterdir()) if d.is_dir()]
            if not runs:
                raise RuntimeError(
                    f"scan_run_subdirectories is set for dataset '{ds['dataset_id']}' "
                    f"but no subdirectories were found in '{ds_input_path}'."
                )
        else:
            if 'runs' not in ds:
                raise KeyError(f"Dataset '{ds['dataset_id']}' is missing required 'runs' field.")
            runs = ds['runs']
        # runs may be a single dict or a list of dicts
        if isinstance(runs, dict):
            runs = [runs]

        for run in runs:
            validate_path_segment(run['run_id'], 'run_id')
            datasets.append({
                'dataset_id':        ds['dataset_id'],
                'run_id':            run['run_id'],
                'exprData':          run.get('exprData', 'ExpressionData.csv'),
                'pseudoTimeData':    run.get('pseudoTimeData', 'PseudoTime.csv'),
                'expressionSource':  run.get('expressionSource'),
                'pseudoTimeSource':  run.get('pseudoTimeSource'),
                'selectedCellsFile': run.get('selectedCellsFile'),
                'groundTruthNetwork': ds.get('groundTruthNetwork', 'GroundTruthNetwork.csv'),
            })

    return datasets


def build_runner(
    algo_name,
    image,
    params,
    resources,
    dataset,
    input_settings,
    output_settings,
):
    if algo_name not in RUNNERS:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Available: {list(RUNNERS)}")

    runner_config = {
        'input': {
            'input_dir': input_settings['input_dir'],
        },
        'dataset': {
            'dataset_id':          dataset['dataset_id'],
            'run_id':              dataset['run_id'],
            'exprData':            dataset['exprData'],
            'pseudoTimeData':      dataset['pseudoTimeData'],
            'expressionSource':    dataset.get('expressionSource'),
            'pseudoTimeSource':    dataset.get('pseudoTimeSource'),
            'selectedCellsFile':   dataset.get('selectedCellsFile'),
            'groundTruthNetwork':  dataset['groundTruthNetwork'],
        },
        'output_settings': {
            'output_dir':      output_settings['output_dir'],
            'experiment_id':   output_settings.get('experiment_id', ''),
        },
        'algo_name': algo_name,
        'image': image,
        'params': params,
        'resources': resources or {},
    }

    return RUNNERS[algo_name](Path.cwd(), runner_config)


def build_runners(config):
    get_working_dirs(config)  # Validate the whole batch before any cleanup.
    input_settings  = config['input_settings']
    output_settings = config['output_settings']
    datasets        = get_datasets(input_settings)
    algorithms      = input_settings.get('algorithms', [])

    runners = []
    for dataset in datasets:
        for algo in algorithms:
            if not enabled(algo.get('should_run')):
                continue
            params = algo.get('params', {})
            setup_started = time.perf_counter()
            runner = build_runner(
                algo['algorithm_id'],
                algo['image'],
                params,
                algo.get('resources', {}),
                dataset,
                input_settings,
                output_settings,
            )
            runner._phase_setup_seconds = time.perf_counter() - setup_started
            runners.append(runner)
    return runners


def get_working_dirs(config):
    """
    Compute expected working_dir paths from config without constructing Runner objects.

    Mirrors the path logic in Runner.__init__ so the overwrite check can run
    before runners are built (and their __init__ erases the directories).

    Parameters
    ----------
    config : dict
        Parsed YAML configuration dictionary.

    Returns
    -------
    list of Path
        One working_dir path per enabled dataset/algorithm combination.
    """
    root            = Path.cwd()
    input_settings  = config['input_settings']
    output_settings = config['output_settings']
    experiment_id   = output_settings.get('experiment_id', '')
    output_dir      = Path(output_settings['output_dir'])
    validate_path_segment(experiment_id, 'experiment_id', allow_empty=True)
    datasets        = get_datasets(input_settings)
    algorithms      = input_settings.get('algorithms', [])

    paths = []
    for dataset in datasets:
        for algo in algorithms:
            if not enabled(algo.get('should_run')):
                continue
            validate_path_segment(algo['algorithm_id'], 'algorithm_id')
            if algo['algorithm_id'] not in RUNNERS:
                raise ValueError('Unknown algorithm: ' + algo['algorithm_id'])
            base_output = output_dir if output_dir.is_absolute() else root / output_dir
            if experiment_id:
                base_output = base_output / experiment_id
            base_output = base_output / dataset['dataset_id'] / dataset['run_id'] / algo['algorithm_id']
            path = (base_output / 'working_dir').resolve()
            if path in paths:
                raise ValueError(f'Duplicate output directory in configuration: {path}')
            paths.append(path)

    return paths


def warn_if_populated(working_dirs):
    """
    Warn and prompt the user if any working directory already contains files.

    Parameters
    ----------
    working_dirs : list of Path
        Working directory paths to check for existing content.

    Returns
    -------
    bool
        True if the user confirms they want to proceed, False otherwise.
    """
    n_populated = sum(
        1 for p in working_dirs
        if p.exists() and any(p.iterdir())
    )
    if n_populated == 0:
        return True

    answer = input(
        f"Warning: {n_populated} working director{'y' if n_populated == 1 else 'ies'} "
        f"already exist and will be overwritten. Proceed? [y/n]: "
    ).strip().lower()
    return answer == 'y'


def execute_config(config_path, *, overwrite=False, interactive=True):
    config = load_config(config_path)
    working_dirs = get_working_dirs(config)
    populated = any(path.exists() and any(path.iterdir()) for path in working_dirs)
    if populated and not overwrite:
        if not interactive or not warn_if_populated(working_dirs):
            raise RuntimeError('Run aborted because a working directory is populated; set overwrite explicitly to retry.')

    runners = build_runners(config)

    for runner in tqdm(runners):
        tqdm.write(runner.running_message)
        execute_runner(runner)


def _write_worker_response(response_path, payload):
    response_path = Path(response_path)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = response_path.with_suffix(response_path.suffix + '.tmp')
    temporary_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary_path.replace(response_path)


def run_worker(ready_file=None):
    """Process configs sequentially without repeating BLRunner imports/startup."""
    if ready_file:
        _write_worker_response(
            ready_file,
            {'status': 'Ready', 'pid': os.getpid()},
        )
    try:
        for request_line in sys.stdin:
            request_line = request_line.strip()
            if not request_line:
                continue

            request_started = time.perf_counter()
            request_id = None
            response_path = None
            try:
                request = json.loads(request_line)
                request_id = request.get('request_id')
                response_path = request['response_path']
                overwrite = request.get('overwrite', False)
                if not isinstance(overwrite, bool):
                    raise ValueError('overwrite must be a boolean.')
                execute_config(request['config_path'], overwrite=overwrite, interactive=False)
                response = {
                    'request_id': request_id,
                    'status': 'Completed',
                    'elapsed_seconds': _rounded_seconds(
                        time.perf_counter() - request_started
                    ),
                }
            except Exception as exc:
                traceback.print_exc()
                response = {
                    'request_id': request_id,
                    'status': 'Failed',
                    'error_type': type(exc).__name__,
                    'error_message': str(exc),
                    'elapsed_seconds': _rounded_seconds(
                        time.perf_counter() - request_started
                    ),
                }

            if response_path:
                try:
                    _write_worker_response(response_path, response)
                except Exception:
                    traceback.print_exc()
    finally:
        shutdown_celloracle_workers()


def main():
    args = parse_args()
    if args.worker:
        run_worker(args.ready_file)
        return
    if not args.config:
        raise SystemExit('--config is required unless --worker is used.')
    execute_config(args.config, overwrite=args.overwrite, interactive=sys.stdin.isatty())


if __name__ == '__main__':
    main()
