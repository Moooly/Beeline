from __future__ import annotations

import atexit
import csv
import heapq
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time
import uuid
import pandas as pd

from BLRun.runner import Runner


_CELLORACLE_WORKERS = {}
_CELLORACLE_WORKERS_LOCK = threading.Lock()


def _celloracle_script_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "Algorithms"
        / "CELLORACLE"
        / "runCellOracle.py"
    )


def _runtime_mount_root(output_dir: Path) -> Path:
    """Choose a scope-level mount shared by every confidence run."""
    resolved = output_dir.resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name == "outputs":
            return candidate.parent
    # Standalone BEELINE layout: <output>/<dataset>/<run>/<algorithm>.
    return resolved.parents[2]


class _CellOracleDockerWorker:
    def __init__(
        self,
        *,
        image: str,
        mount_root: Path,
        cpu_budget: int,
        memory_budget_mb: int | None,
    ) -> None:
        self.image = image
        self.mount_root = mount_root.resolve()
        self.cpu_budget = max(1, int(cpu_budget))
        self.memory_budget_mb = (
            max(512, int(memory_budget_mb))
            if memory_budget_mb is not None
            else None
        )
        self.process = None
        self.log_file = None
        self.lock = threading.Lock()

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return

        self.mount_root.mkdir(parents=True, exist_ok=True)
        script_path = _celloracle_script_path()
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            f"--cpus={self.cpu_budget}",
            "-e",
            f"OMP_NUM_THREADS={self.cpu_budget}",
            "-e",
            f"OPENBLAS_NUM_THREADS={self.cpu_budget}",
            "-e",
            f"MKL_NUM_THREADS={self.cpu_budget}",
            "-e",
            f"NUMEXPR_NUM_THREADS={self.cpu_budget}",
            "-e",
            f"VECLIB_MAXIMUM_THREADS={self.cpu_budget}",
            "-e",
            f"NUMBA_NUM_THREADS={self.cpu_budget}",
        ]
        if self.memory_budget_mb is not None:
            command.append(f"--memory={self.memory_budget_mb}m")
        command.extend(
            [
                "-v",
                f"{self.mount_root}:/grnscope/runtime",
                "-v",
                f"{script_path}:/runCellOracle.py:ro",
                self.image,
                "python",
                "/runCellOracle.py",
                "--worker",
            ]
        )
        self.log_file = (self.mount_root / "celloracle-worker.log").open(
            "a", encoding="utf-8"
        )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def _container_path(self, host_path: Path) -> str:
        relative_path = host_path.resolve().relative_to(self.mount_root)
        return str(Path("/grnscope/runtime") / relative_path)

    def submit(self, runner) -> dict:
        with self.lock:
            self.start()
            if self.process is None or self.process.stdin is None:
                raise RuntimeError("CellOracle persistent container did not start.")
            if self.process.poll() is not None:
                raise RuntimeError(
                    "CellOracle persistent container exited with code "
                    f"{self.process.returncode}."
                )

            response_path = runner.working_dir / (
                f"celloracle-response-{uuid.uuid4().hex}.json"
            )
            try:
                response_path.unlink()
            except FileNotFoundError:
                pass
            argv = runner._inference_arguments(
                in_file=self._container_path(
                    runner.working_dir / "ExpressionData.csv"
                ),
                out_file=self._container_path(runner.working_dir / "outFile.txt"),
            )
            request = {
                "request_id": response_path.stem,
                "argv": argv,
                "responseFile": self._container_path(response_path),
                "logFile": self._container_path(runner.output_dir / "output.txt"),
                "timeFile": self._container_path(runner.working_dir / "time.txt"),
            }
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
            except BrokenPipeError as exc:
                raise RuntimeError(
                    "CellOracle persistent container closed unexpectedly."
                ) from exc

            while not response_path.is_file():
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "CellOracle persistent container exited before returning "
                        f"a result (code {self.process.returncode})."
                    )
                time.sleep(0.1)

            try:
                response = json.loads(response_path.read_text(encoding="utf-8"))
            finally:
                try:
                    response_path.unlink()
                except FileNotFoundError:
                    pass
            if response.get("status") != "Completed":
                raise RuntimeError(
                    response.get("error_message")
                    or "CellOracle persistent inference failed."
                )
            return response

    def stop(self, *, force: bool = False) -> None:
        process = self.process
        if process is None:
            return
        if process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self.log_file is not None and not self.log_file.closed:
            self.log_file.close()
        self.process = None


def _get_celloracle_worker(runner):
    mount_root = _runtime_mount_root(runner.output_dir)
    key = (
        runner.image,
        str(mount_root),
        int(getattr(runner, "cpu_budget", 1)),
        getattr(runner, "memory_budget_mb", None),
    )
    with _CELLORACLE_WORKERS_LOCK:
        worker = _CELLORACLE_WORKERS.get(key)
        if worker is None:
            worker = _CellOracleDockerWorker(
                image=runner.image,
                mount_root=mount_root,
                cpu_budget=getattr(runner, "cpu_budget", 1),
                memory_budget_mb=getattr(runner, "memory_budget_mb", None),
            )
            _CELLORACLE_WORKERS[key] = worker
        return worker


def shutdown_celloracle_workers() -> None:
    with _CELLORACLE_WORKERS_LOCK:
        workers = list(_CELLORACLE_WORKERS.values())
        _CELLORACLE_WORKERS.clear()
    for worker in workers:
        worker.stop()


atexit.register(shutdown_celloracle_workers)


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
            ExpressionData = self.read_expression_data()
            ExpressionData.to_csv(CELLORACLE_EXPRESSION_FILE,
                                  sep=',', header=True, index=True)

    def run(self):
        '''
        Function to run CellOracle inside the Docker image.
        '''
        # Parameter tests and third-party standalone runners may inject their
        # own Docker executor. Preserve that compatibility with the one-shot
        # command while GRNScope uses the persistent container path below.
        if '_run_docker' in self.__dict__:
            self._run_legacy_container()
            return

        response = _get_celloracle_worker(self).submit(self)
        self.persistent_runtime_observability = {
            "kind": "celloracle_container_worker",
            "worker_pid": response.get("worker_pid"),
            "request_index": response.get("request_index"),
            "prior_cache_hit": response.get("prior_cache_hit"),
            "preprocessing_mode": response.get("preprocessing_mode"),
            "worker_elapsed_seconds": response.get("elapsed_seconds"),
            "edge_count": response.get("edge_count"),
        }

    def _inference_arguments(self, *, in_file: str, out_file: str) -> list[str]:
        requested_top_k = self.params.get('topK', 0)
        output_top_k = self.params.get('maxRegulatorsPerTarget', 0)
        positive_limits = []
        for raw_limit in (requested_top_k, output_top_k):
            try:
                parsed_limit = int(raw_limit)
            except (TypeError, ValueError):
                continue
            if parsed_limit > 0:
                positive_limits.append(parsed_limit)
        effective_top_k = min(positive_limits) if positive_limits else 0
        return [
            '--inFile', in_file,
            '--outFile', out_file,
            '--species', str(self.params.get('species', 'human')),
            '--baseGrn', str(self.params.get('baseGrn', 'auto')),
            '--alpha', str(self.params.get('alpha', 10)),
            '--pValueCutoff', str(self.params.get('pValueCutoff', 0.05)),
            '--topK', str(effective_top_k),
            '--maxGenes', str(self.params.get('maxGenes', 0)),
            '--maxCells', str(self.params.get('maxCells', 0)),
            '--nJobs', str(getattr(self, 'cpu_budget', 1)),
        ]

    def _run_legacy_container(self):
        arguments = self._inference_arguments(
            in_file='/usr/working_dir/ExpressionData.csv',
            out_file='/usr/working_dir/outFile.txt',
        )
        argument_text = ' '.join(shlex.quote(value) for value in arguments)
        script_path = shlex.quote(str(_celloracle_script_path()))
        working_dir = shlex.quote(str(self.working_dir))

        cmdToRun = ' '.join(['docker run --rm',
                            f"-v {working_dir}:/usr/working_dir",
                            f"-v {script_path}:/runCellOracle.py:ro",
                            f'{self.image} /bin/sh -c \"time -v -o',
                            "/usr/working_dir/time.txt",
                            'python /runCellOracle.py',
                            argument_text, '\"'])

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
