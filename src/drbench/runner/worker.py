"""Run one (dataset, algorithm, seed) in a fresh process, which isolates its memory and imports.

The parent sets the thread environment variables before this interpreter starts. The worker
pins its CPU affinity, loads the data, runs an untimed warm-up on a small slice (so JIT
compilation is not timed), resets the kernel's peak-RSS counter, then times `fit_transform` on the
full data. The embedding is written to `.npy`; timing and memory go to a sidecar `.json`.
CPU time (user + system, summed over all threads) is measured alongside wall time, because
implementations differ in how well they use the available cores.
Peak RSS still includes everything resident at the start: interpreter, data, loaded libraries.
"""

import argparse
import json
import os
import resource
import time
from pathlib import Path

from drbench.runner import memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--warmup-rows", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-path", type=Path, required=True)
    return parser.parse_args()


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def main() -> None:
    args = parse_args()
    os.sched_setaffinity(0, range(args.threads))

    import numpy as np

    from drbench import algorithms, datasets

    algorithm = algorithms.get(args.algorithm)
    X = datasets.load(args.dataset, args.data_dir)
    if args.warmup_rows:
        algorithm.fit_transform(X[: args.warmup_rows].copy(), args.seed, args.threads, args.budget)
    memory.reset_peak_rss()
    baseline_kb = memory.current_rss_kb()

    start, cpu_start = time.perf_counter(), cpu_seconds()
    Y = np.asarray(algorithm.fit_transform(X, args.seed, args.threads, args.budget), dtype=np.float32)
    fit_s, cpu_s = time.perf_counter() - start, cpu_seconds() - cpu_start
    peak_kb = memory.peak_rss_kb()

    args.embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.embedding_path, Y)
    sidecar = {
        "fit_s": fit_s, "cpu_s": cpu_s, "baseline_rss_kb": baseline_kb, "peak_rss_kb": peak_kb,
        "n": len(X), "dim": X.shape[1],
    }
    args.embedding_path.with_suffix(".json").write_text(json.dumps(sidecar))


if __name__ == "__main__":
    main()
