"""Aggregate run and metric records into one row per (dataset, algorithm)."""

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drbench import algorithms
from drbench.runner.records import RunKey, latest_by_key, metrics_path, read_jsonl, runs_path

COST_FIELDS = ("fit_s", "peak_rss_mb", "delta_rss_mb")
KEY_FIELDS = ("dataset", "algorithm", "seed")


@dataclass
class Summary:
    dataset: str
    algorithm: str
    n_ok: int = 0
    n_failed: int = 0
    cost: dict[str, float] = field(default_factory=dict)
    score_mean: dict[str, float] = field(default_factory=dict)
    score_sd: dict[str, float] = field(default_factory=dict)

    @property
    def family(self) -> str:
        return algorithms.get(self.algorithm).family

    @property
    def library(self) -> str:
        return algorithms.get(self.algorithm).library


def score_names(metric_records: list[dict]) -> list[str]:
    excluded = set(KEY_FIELDS) | {"eval_s"}
    return [name for name in metric_records[0] if name not in excluded] if metric_records else []


def summarize(results_dir: Path, datasets: tuple[str, ...]) -> list[Summary]:
    runs = latest_by_key(read_jsonl(runs_path(results_dir)))
    metrics = latest_by_key(read_jsonl(metrics_path(results_dir)))
    names = score_names(list(metrics.values()))

    grouped: dict[tuple[str, str], list[RunKey]] = defaultdict(list)
    for key in sorted(runs):
        if key.dataset in datasets:
            grouped[(key.dataset, key.algorithm)].append(key)

    summaries = []
    for (dataset, algorithm), keys in grouped.items():
        ok = [k for k in keys if runs[k]["status"] == "ok"]
        summary = Summary(dataset, algorithm, n_ok=len(ok), n_failed=len(keys) - len(ok))
        summary.cost = {f: float(np.median([runs[k][f] for k in ok])) for f in COST_FIELDS} if ok else {}
        scored = [metrics[k] for k in ok if k in metrics]
        for name in names if scored else ():
            values = np.array([record[name] for record in scored])
            summary.score_mean[name] = float(values.mean())
            summary.score_sd[name] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summaries.append(summary)
    return sorted(summaries, key=lambda s: (s.dataset, -s.score_mean.get("local", -1.0)))


def write_csv(summaries: list[Summary], path: Path) -> None:
    names = sorted({name for s in summaries for name in s.score_mean})
    header = ["dataset", "algorithm", "family", "library", "n_ok", "n_failed", *COST_FIELDS,
              *[f"{n}_mean" for n in names], *[f"{n}_sd" for n in names]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(header)
        for s in summaries:
            writer.writerow([
                s.dataset, s.algorithm, s.family, s.library, s.n_ok, s.n_failed,
                *[s.cost.get(f, "") for f in COST_FIELDS],
                *[s.score_mean.get(n, "") for n in names], *[s.score_sd.get(n, "") for n in names],
            ])
