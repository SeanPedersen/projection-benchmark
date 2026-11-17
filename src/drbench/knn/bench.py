"""Time each kNN backend on each dataset and score its recall against the exact graph.

Resumable: a finished `(dataset, backend, k, repeat)` is skipped. Like the embedding sweep, the
process is pinned to `n_threads` cores and must not share the machine with another timed run.
"""

import os
import time
from dataclasses import dataclass
from importlib import metadata

import numpy as np

from drbench import datasets
from drbench.config import Config
from drbench.knn import backends
from drbench.runner.orchestrator import thread_environment
from drbench.runner.records import append_jsonl, read_jsonl

RECALL_CHUNK_ROWS = 10_000
EXACT_BACKEND = "mrs-exhaustive"


@dataclass(frozen=True)
class KnnKey:
    dataset: str
    backend: str
    k: int
    repeat: int


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of the exact k neighbours that `found` contains, averaged over all rows."""
    hits = 0
    for start in range(0, len(truth), RECALL_CHUNK_ROWS):
        rows = slice(start, start + RECALL_CHUNK_ROWS)
        hits += int((found[rows, :, None] == truth[rows, None, :]).any(axis=1).sum())
    return hits / truth.size


def exact_neighbors(X: np.ndarray, k: int, cache_dir, n_threads: int) -> np.ndarray:
    """Exact kNN indices with at least `k` columns, cached next to the dataset."""
    cache = cache_dir / "knn_exact.npy"
    if cache.exists() and (cached := np.load(cache)).shape[1] >= k:
        return cached[:, :k]
    truth = backends.get(EXACT_BACKEND).build(X, k, n_threads)
    np.save(cache, truth)
    return truth


def pin_process(n_threads: int) -> None:
    os.environ.update(thread_environment(n_threads))
    os.sched_setaffinity(0, range(n_threads))


def library_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def time_build(backend: backends.KnnBackend, X: np.ndarray, k: int, n_threads: int):
    started = time.perf_counter()
    found = backend.build(X, k, n_threads)
    return np.asarray(found), time.perf_counter() - started


def bench_backend(backend, X, truth, key_base, config, store) -> None:
    knn = config.knn
    for k in knn.ks:
        for repeat in range(knn.repeats):
            key = KnnKey(key_base, backend.name, k, repeat)
            record = key.__dict__ | {
                "n": len(X), "dim": X.shape[1], "n_threads": config.run.n_threads,
                "library_version": library_version(backend.package),
            }
            try:
                found, seconds = time_build(backend, X, k, config.run.n_threads)
                record |= {"status": "ok", "time_s": seconds, "recall": recall_at_k(found, truth[:, :k])}
            except Exception as error:  # noqa: BLE001 - a failing backend must not stop the sweep
                record |= {"status": "failed", "error": repr(error)}
            append_jsonl(store, record)
            print(f"  {backend.name:16s} k={k:3d} #{repeat}: "
                  f"{record.get('time_s', float('nan')):7.1f}s recall={record.get('recall', float('nan')):.4f}",
                  flush=True)


def finished_keys(store) -> set[KnnKey]:
    return {
        KnnKey(r["dataset"], r["backend"], r["k"], r["repeat"])
        for r in read_jsonl(store) if r["status"] == "ok"
    }


def run_knn(config: Config, dataset_specs: tuple[str, ...], backend_names: tuple[str, ...]) -> None:
    knn, n_threads = config.knn, config.run.n_threads
    pin_process(n_threads)
    store = knn.results_dir / "knn.jsonl"
    for spec in dataset_specs:
        X = datasets.load(spec, config.run.data_dir)
        print(f"{spec}: n={len(X)} dim={X.shape[1]}", flush=True)
        truth = exact_neighbors(X, max(knn.ks), datasets.cache_dir(spec, config.run.data_dir), n_threads)
        for name in backend_names:
            backend = backends.get(name)
            todo = [(k, r) for k in knn.ks for r in range(knn.repeats)
                    if KnnKey(spec, name, k, r) not in finished_keys(store)]
            if not todo:
                continue
            warmup = X[: min(knn.warmup_rows, len(X))]
            backend.build(warmup, min(knn.ks[0], len(warmup) - 1), n_threads)  # untimed, absorbs JIT
            bench_backend(backend, X, truth, spec, config, store)
