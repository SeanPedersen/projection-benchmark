"""Score every successful run that has no current metrics yet; results go to `metrics.jsonl`.
Records scored under an older `METRICS_VERSION` are rescored (appended; the latest wins).

Evaluation is never timed as part of a run. The original-space reference (exact kNN, k-means,
pairs) is built once per dataset spec and cached.
"""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from drbench import datasets
from drbench.config import Config
from drbench.metrics import METRICS_VERSION, evaluate, reference_for
from drbench.runner.records import (
    RunKey,
    append_jsonl,
    latest_by_key,
    metrics_path,
    read_jsonl,
    runs_path,
)


def current_scores(results_dir: Path) -> dict[RunKey, dict]:
    """Latest metric record per run, if it was scored under the current definitions and belongs
    to the latest run of that key (same epochs; a rerun with a new schedule needs new scores)."""
    runs = latest_by_key(read_jsonl(runs_path(results_dir)))
    latest = latest_by_key(read_jsonl(metrics_path(results_dir)))
    return {key: record for key, record in latest.items()
            if record.get("metrics_version") == METRICS_VERSION and key in runs
            and record.get("epochs") == runs[key].get("epochs")}


def unevaluated_runs(config: Config) -> dict[str, list[RunKey]]:
    results_dir = config.run.results_dir
    runs = latest_by_key(read_jsonl(runs_path(results_dir)))
    scored = set(current_scores(results_dir))
    by_dataset: dict[str, list[RunKey]] = defaultdict(list)
    for key, record in sorted(runs.items()):
        if record["status"] == "ok" and key not in scored and key.dataset in config.run.datasets:
            by_dataset[key.dataset].append(key)
    return by_dataset


def evaluate_pending(config: Config) -> None:
    run, metrics = config.run, config.metrics
    runs = latest_by_key(read_jsonl(runs_path(run.results_dir)))
    for spec, keys in unevaluated_runs(config).items():
        X = datasets.load(spec, run.data_dir)
        started = time.perf_counter()
        reference = reference_for(X, datasets.cache_dir(spec, run.data_dir) / "reference.npz", metrics)
        print(f"{spec}: reference ready ({time.perf_counter() - started:.0f}s)", flush=True)
        for key in keys:
            started = time.perf_counter()
            Y = np.load(key.embedding_path(run.results_dir))
            scores = evaluate(X, Y, reference, metrics)
            record = key.as_dict() | scores | {"epochs": runs[key].get("epochs"),
                                                 "eval_s": time.perf_counter() - started}
            append_jsonl(metrics_path(run.results_dir), record)
            print(f"  {key.algorithm} seed={key.seed} ×{key.budget:g}: local={scores['local']:.4f} "
                  f"global={scores['global']:.4f} ({record['eval_s']:.0f}s)", flush=True)
