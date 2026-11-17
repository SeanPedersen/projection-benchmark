"""Run the dataset × algorithm × seed sweep, one isolated worker process per run.

- Fair compute budget: every worker gets the same thread count through the usual thread-pool
  environment variables and the same CPU affinity.
- Peak memory: measured by the worker (VmHWM of the timed phase, see `memory.py`). The
  `ru_maxrss` from `os.wait4` is kept only as an upper bound for runs that crash or time out.
- Robustness: a per-run timeout. Failures are recorded, not fatal. Runs already
  completed in `runs.jsonl` are skipped, unless their recorded epochs differ from the current
  schedule (then they are rerun and the new record wins).
- Thermal fairness on laptops: the run order is shuffled (fixed seed), with an optional cooldown.
"""

import itertools
import json
import os
import random
import signal
import subprocess
import sys
import time
from importlib import metadata

from drbench import algorithms
from drbench.config import Config
from drbench.runner.records import RunKey, append_jsonl, latest_by_key, read_jsonl, runs_path

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS",
)
ORDER_SEED = 0
POLL_INTERVAL_S = 0.2
KB_PER_MB = 1024


def shuffled(keys: list[RunKey]) -> list[RunKey]:
    order = list(keys)
    random.Random(ORDER_SEED).shuffle(order)
    return order


def default_plan(config: Config) -> list[RunKey]:
    """Every algorithm at its library-default budget, for every seed."""
    run = config.run
    return shuffled([RunKey(*combo) for combo in itertools.product(run.datasets, run.algorithms, run.seeds)])


def is_stale(key: RunKey, record: dict) -> bool:
    """A successful run whose epoch count no longer matches the algorithm's schedule."""
    return record.get("epochs") != algorithms.get(key.algorithm).epochs(key.budget)


def pending_runs(keys: list[RunKey], config: Config, retry_failed: bool) -> list[RunKey]:
    previous = latest_by_key(read_jsonl(runs_path(config.run.results_dir)))
    def is_done(key: RunKey) -> bool:
        record = previous.get(key)
        if record is None:
            return False
        if record["status"] == "ok":
            return not is_stale(key, record)
        return not retry_failed
    return [key for key in keys if not is_done(key)]


def thread_environment(n_threads: int) -> dict[str, str]:
    return os.environ | {name: str(n_threads) for name in THREAD_ENV_VARS}


def worker_command(key: RunKey, config: Config) -> list[str]:
    run = config.run
    return [
        sys.executable, "-m", "drbench.runner.worker",
        "--dataset", key.dataset, "--algorithm", key.algorithm, "--seed", str(key.seed),
        "--threads", str(run.n_threads), "--budget", str(key.budget), "--warmup-rows", str(run.warmup_rows),
        "--data-dir", str(run.data_dir),
        "--embedding-path", str(key.embedding_path(run.results_dir)),
    ]


def wait_with_timeout(process: subprocess.Popen, timeout_s: float) -> tuple[int, int, bool]:
    """Return (exit status, peak RSS in KB, timed out) using wait4 for the child's rusage."""
    deadline = time.monotonic() + timeout_s
    while True:
        pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if pid:
            process.returncode = os.waitstatus_to_exitcode(status)
            return process.returncode, usage.ru_maxrss, False
        if time.monotonic() > deadline:
            process.send_signal(signal.SIGKILL)
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            return process.returncode, usage.ru_maxrss, True
        time.sleep(POLL_INTERVAL_S)


def library_version(algorithm_name: str) -> str:
    try:
        return metadata.version(algorithms.get(algorithm_name).library)
    except metadata.PackageNotFoundError:
        return "unknown"


def execute(key: RunKey, config: Config) -> dict:
    run = config.run
    embedding_path = key.embedding_path(run.results_dir)
    sidecar_path = embedding_path.with_suffix(".json")
    sidecar_path.unlink(missing_ok=True)
    log_path = key.log_path(run.results_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            worker_command(key, config), env=thread_environment(run.n_threads),
            stdout=log, stderr=subprocess.STDOUT,
        )
        exit_code, peak_kb, timed_out = wait_with_timeout(process, run.timeout_s)
    wall_s = time.perf_counter() - started

    record = key.as_dict() | {
        "status": "timeout" if timed_out else ("ok" if exit_code == 0 and sidecar_path.exists() else "failed"),
        "exit_code": exit_code,
        "wall_s": wall_s,
        "peak_rss_mb": peak_kb / KB_PER_MB,
        "peak_rss_source": "wait4",
        "n_threads": run.n_threads,
        "library_version": library_version(key.algorithm),
        "timeout_s": run.timeout_s,
    }
    if record["status"] != "ok":
        return record
    sidecar = json.loads(sidecar_path.read_text())
    peak_mb = sidecar["peak_rss_kb"] / KB_PER_MB
    baseline_mb = sidecar["baseline_rss_kb"] / KB_PER_MB
    return record | {
        "fit_s": sidecar["fit_s"],
        "cpu_s": sidecar["cpu_s"],
        "shared_knn_s": sidecar.get("shared_knn_s", 0.0),
        "epochs": algorithms.get(key.algorithm).epochs(key.budget),
        "peak_rss_mb": peak_mb,
        "peak_rss_source": "worker",
        "baseline_rss_mb": baseline_mb,
        "delta_rss_mb": peak_mb - baseline_mb,
        "n": sidecar["n"],
        "dim": sidecar["dim"],
    }


def run_sweep(config: Config, plan: list[RunKey], retry_failed: bool = False) -> None:
    keys = pending_runs(plan, config, retry_failed)
    for index, key in enumerate(keys, start=1):
        print(f"[{index}/{len(keys)}] {key.dataset} {key.algorithm} seed={key.seed} "
              f"budget=×{key.budget:g}", flush=True)
        record = execute(key, config)
        append_jsonl(runs_path(config.run.results_dir), record)
        fit = f"{record['fit_s']:.1f}s" if "fit_s" in record else "-"
        print(f"    {record['status']}  fit={fit}  peak={record['peak_rss_mb']:.0f}MB", flush=True)
        if config.run.cooldown_s and index < len(keys):
            time.sleep(config.run.cooldown_s)
