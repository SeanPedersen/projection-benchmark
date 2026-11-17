"""Shared RINND kNN graphs for the `-rinnd-fast` variants.

`rinnd_graph(X, k, n_threads)` returns `(indices, distances)` of shape (n, k): plain euclidean
distances, sorted ascending, without the point itself. Each adapter converts that to what its
library expects.

The RINND schedule is chosen per k (`RINND_FAST`), from sweeps on 100k x 512:
- k=15: the defaults (0.73 s at 36 threads, recall 0.894). Cheaper schedules save at most 0.25 s
  and drop recall to 0.72-0.88.
- k=60 and k=90: `max_candidates=20, n_iters=4`. At k=60 (PaCMAP's `n_near + 50`) 1.6 s and recall
  0.936 vs 3.1 s and 0.991 (36 threads); at k=90 9 s and 0.898 vs 31 s and 0.994 (8 threads),
  close to manifolds-rs IVF's 0.883.

Sharing: after its untimed warm-up the worker calls `enable_cache(dir)`. From then on a graph is
built at most once per (dataset, k, schedule, threads) and stored with its build time and its
peak RSS above the worker's baseline. A run that loads a stored graph is charged both
(`charged_cost()`): the build time is added to its fit time, and its peak becomes at least its own
baseline plus the graph's extra memory. So it reports what building the graph itself would cost.
Every run that uses a graph (built or loaded) also reports the graph's build time as `knn_s`.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drbench.runner import memory

GRAPH_SEED = 0
RINND_FAST: dict[int, dict] = {
    15: {},
    60: {"max_candidates": 20, "n_iters": 4},
    90: {"max_candidates": 20, "n_iters": 4},
}


@dataclass
class ChargedCost:
    """`build_s`/`extra_rss_kb`: what loading stored graphs adds on top of the measured run.
    `knn_s`: build time of every graph the run used, built here or loaded (None if it used none),
    so layout time = fit time - knn_s."""

    build_s: float = 0.0
    extra_rss_kb: int = 0
    knn_s: float | None = None

    def add_graph_time(self, seconds: float) -> None:
        self.knn_s = (self.knn_s or 0.0) + seconds


_cache_dir: Path | None = None
_baseline_rss_kb = 0
_charged = ChargedCost()


def enable_cache(directory: Path, baseline_rss_kb: int) -> None:
    global _cache_dir, _baseline_rss_kb
    _cache_dir, _baseline_rss_kb = directory, baseline_rss_kb


def charged_cost() -> ChargedCost:
    return _charged


def schedule_for(k: int) -> dict:
    if k not in RINND_FAST:
        raise KeyError(f"no RINND schedule for k={k}; known: {sorted(RINND_FAST)}")
    return RINND_FAST[k]


def build_rinnd_graph(X: np.ndarray, k: int, n_threads: int) -> tuple[np.ndarray, np.ndarray]:
    from rinnd import RINND

    index = RINND(np.ascontiguousarray(X, dtype=np.float32), n_neighbors=k, n_jobs=n_threads,
                  random_state=GRAPH_SEED, graph_only=True, **schedule_for(k))
    indices, distances = index.neighbor_graph
    return indices.astype(np.int64), distances


def graph_path(directory: Path, k: int, n_points: int, n_threads: int) -> Path:
    schedule = "-".join(f"{key}{value}" for key, value in sorted(schedule_for(k).items())) or "default"
    return directory / f"rinnd-k{k}-{schedule}-n{n_points}-t{n_threads}.npz"


def rinnd_graph(X: np.ndarray, k: int, n_threads: int) -> tuple[np.ndarray, np.ndarray]:
    if _cache_dir is None:
        return build_rinnd_graph(X, k, n_threads)
    path = graph_path(_cache_dir, k, len(X), n_threads)
    if path.exists():
        stored = np.load(path)
        _charged.build_s += float(stored["build_s"])
        _charged.add_graph_time(float(stored["build_s"]))
        _charged.extra_rss_kb = max(_charged.extra_rss_kb, int(stored["extra_rss_kb"]))
        return stored["indices"], stored["distances"]
    started = time.perf_counter()
    indices, distances = build_rinnd_graph(X, k, n_threads)
    build_s = time.perf_counter() - started
    _charged.add_graph_time(build_s)
    path.parent.mkdir(parents=True, exist_ok=True)
    # VmHWM is the peak since the worker's reset, so this is what the build added over baseline.
    np.savez(path, indices=indices, distances=distances, build_s=build_s,
             extra_rss_kb=memory.peak_rss_kb() - _baseline_rss_kb)
    return indices, distances


def with_self(indices: np.ndarray, distances: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """umap-learn layout: each point is its own first neighbour, k columns in total."""
    rows = np.arange(len(indices), dtype=indices.dtype)[:, None]
    return (np.hstack([rows, indices[:, : k - 1]]),
            np.hstack([np.zeros_like(distances[:, :1]), distances[:, : k - 1]]))
