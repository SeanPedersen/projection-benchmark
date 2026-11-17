"""Scaling report: seed aggregation, the relative combined score, ranked tables and the map."""

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
# local, global per algorithm: pca has the best global, umap the best local.
SCORES = {"pca": (0.6, 0.30), "mrs-umap-rinnd-default": (0.9, 0.10),
          "ncvis-rinnd-default": (0.8, 0.20)}


def write_records(results_dir: Path) -> None:
    for n in SIZES:
        for algorithm, (local, global_rho) in SCORES.items():
            for seed in (0, 1):
                key = {"dataset": f"dbpedia-512-100k@{n}", "algorithm": algorithm, "seed": seed,
                       "budget": 1.0}
                append_jsonl(runs_path(results_dir), key | {
                    "status": "ok", "fit_s": n / 1000 + seed, "knn_s": None,
                    "peak_rss_mb": 1024.0, "n": n, "epochs": None,
                })
                append_jsonl(metrics_path(results_dir), key | {
                    "metrics_version": METRICS_VERSION, "epochs": None, "local": local,
                    "global": global_rho + seed / 100, "triplet_acc": 0.6,
                })


def test_scaling_report_ranks_by_combined_score(tmp_path):
    write_records(tmp_path)
    config = load_config(Path("benchmark.toml")).with_run(
        datasets=tuple(f"dbpedia-512-100k@{n}" for n in SIZES), algorithms=tuple(SCORES),
        results_dir=tmp_path)
    config = replace(config, run=replace(config.run, seeds=(0, 1)))
    report = write_scaling_report(config).read_text()
    # ncvis: local (0.8 - 0.6) / 0.3 = 2/3, global (0.205 - 0.105) / 0.2 = 1/2 -> 0.58, ranked
    # above pca and mrs-umap (each best on one scale, worst on the other: 0.50).
    row = "| ncvis-rinnd-default | 0.58 | 0.8000 ± 0.0000 | 0.205 ± 0.007 | 0.600 ± 0.000 | 10.5 | 1.0 |"
    assert row in report
    table = report[report.index("## N = 10,000"):]
    assert table.index("ncvis-rinnd-default") < table.index("| pca |")
    assert report.count("## N = ") == len(SIZES)
    assert (tmp_path / "figures" / "dbpedia-512-100k-quality-map.png").stat().st_size


def test_spread_positions_keeps_gap_and_centres_groups():
    assert spread_positions([0.0, 0.5, 1.0], 0.1) == [0.0, 0.5, 1.0]
    desired = [0.50, 0.50, 0.52]
    placed = spread_positions(desired, 0.1)
    assert all(b - a >= 0.1 - 1e-12 for a, b in itertools.pairwise(placed))
    assert sum(placed) == pytest.approx(sum(desired))
