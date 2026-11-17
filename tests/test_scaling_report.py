"""Scaling report: aggregation over seeds, layout time, table and charts from synthetic records."""

import itertools
from dataclasses import replace
from pathlib import Path

import pytest

from drbench.config import load_config
from drbench.metrics import METRICS_VERSION
from drbench.report.plots import spread_positions
from drbench.report.scaling import write_scaling_report
from drbench.runner.records import append_jsonl, metrics_path, runs_path

SIZES = (1000, 10000)
ALGORITHMS = {"pca": None, "mrs-umap-rinnd-fast": 0.5}


def write_records(results_dir: Path) -> None:
    for n in SIZES:
        for algorithm, knn_s in ALGORITHMS.items():
            for seed in (0, 1):
                key = {"dataset": f"dbpedia-512-100k@{n}", "algorithm": algorithm, "seed": seed,
                       "budget": 1.0}
                fit_s = n / 1000 + seed
                append_jsonl(runs_path(results_dir), key | {
                    "status": "ok", "fit_s": fit_s, "knn_s": knn_s and knn_s * n / 1000,
                    "peak_rss_mb": 100.0 + n / 100, "n": n, "epochs": None,
                })
                append_jsonl(metrics_path(results_dir), key | {
                    "metrics_version": METRICS_VERSION, "epochs": None, "local": 0.9,
                    "global": 0.3, "triplet_acc": 0.6 + seed / 100,
                })


def test_scaling_report(tmp_path):
    write_records(tmp_path)
    config = load_config(Path("benchmark.toml")).with_run(
        datasets=tuple(f"dbpedia-512-100k@{n}" for n in SIZES), algorithms=tuple(ALGORITHMS),
        results_dir=tmp_path)
    config = replace(config, run=replace(config.run, seeds=(0, 1)))
    report = write_scaling_report(config).read_text()
    # seeds 0 and 1 at N=10k: fit 10 and 11 s, kNN 5 s -> layout 5.5 ± 0.7
    assert "| mrs-umap-rinnd-fast | 10,000 | 10.5 ± 0.7 | 5.0 ± 0.0 | 5.5 ± 0.7 |" in report
    assert "| pca | 1,000 | 1.5 ± 0.7 | 0.0 ± 0.0 | 1.5 ± 0.7 |" in report
    for chart in ("time", "memory", "quality"):
        assert (tmp_path / "figures" / f"dbpedia-512-100k-scaling-{chart}.png").stat().st_size


def test_spread_positions_keeps_gap_and_centres_groups():
    assert spread_positions([0.0, 0.5, 1.0], 0.1) == [0.0, 0.5, 1.0]
    desired = [0.50, 0.50, 0.52]
    placed = spread_positions(desired, 0.1)
    assert all(b - a >= 0.1 - 1e-12 for a, b in itertools.pairwise(placed))
    assert sum(placed) == pytest.approx(sum(desired))
