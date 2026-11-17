"""Scaling report: projection quality of every algorithm on one dataset at several sizes (`name@N`).

Quality is local structure (mean trustworthiness) and global structure (random-pair ρ) together:
- a map per N (`plot_quality_map`): local vs global, one point per algorithm;
- a table per N, ranked by `combined` = mean of local and global after min-max scaling each across
  the algorithms at that N (1 = best of the compared methods on both). It is relative: it ranks
  methods within one N and changes when the set of methods changes.
Fit time and peak memory stay as side columns. Mean ± sd over seeds.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drbench import algorithms, datasets
from drbench.config import Config
from drbench.report.plots import plot_quality_map
from drbench.runner.evaluate import current_scores
from drbench.runner.records import latest_by_key, read_jsonl, runs_path

FIELDS = ("local", "global", "triplet_acc", "fit_s", "knn_s", "peak_rss_mb")
COMBINED_FROM = ("local", "global")
MB_PER_GB = 1024


@dataclass(frozen=True)
class Point:
    """Mean and sd over seeds of one algorithm at one N."""

    algorithm: str
    n: int
    n_seeds: int
    mean: dict[str, float]
    sd: dict[str, float]

    @property
    def library(self) -> str:
        return algorithms.get(self.algorithm).library


def joined_records(
    results_dir: Path, specs: tuple[str, ...], names: tuple[str, ...]
) -> dict[tuple[str, int], list[dict]]:
    """Successful, scored runs of `names` grouped by (algorithm, N)."""
    runs = latest_by_key(read_jsonl(runs_path(results_dir)))
    scores = current_scores(results_dir)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for key, run in runs.items():
        if (key.dataset not in specs or key.algorithm not in names or run["status"] != "ok"
                or key not in scores):
            continue
        grouped[(key.algorithm, run["n"])].append(
            run | scores[key] | {"knn_s": run.get("knn_s") or 0.0}
        )
    return grouped


def aggregate(grouped: dict[tuple[str, int], list[dict]]) -> list[Point]:
    points = []
    for (algorithm, n), records in sorted(grouped.items()):
        values = {field: np.array([r[field] for r in records]) for field in FIELDS}
        points.append(Point(
            algorithm, n, len(records),
            mean={f: float(v.mean()) for f, v in values.items()},
            sd={f: float(v.std(ddof=1)) if len(v) > 1 else 0.0 for f, v in values.items()},
        ))
    return points


def min_max(values: np.ndarray) -> np.ndarray:
    span = values.max() - values.min()
    return (values - values.min()) / span if span > 0 else np.ones_like(values)


def combined_scores(points: list[Point]) -> dict[str, float]:
    """`combined` per algorithm for points of one N (see module docstring)."""
    scaled = [min_max(np.array([p.mean[field] for p in points])) for field in COMBINED_FROM]
    return {p.algorithm: float(score) for p, score in zip(points, np.mean(scaled, axis=0))}


def with_sd(point: Point, field: str, fmt: str) -> str:
    text = fmt.format(point.mean[field])
    return f"{text} ± {fmt.format(point.sd[field])}" if point.n_seeds > 1 else text


def quality_table(points: list[Point]) -> str:
    """One N: ranked by combined score; time and memory as side columns."""
    combined = combined_scores(points)
    header = ["algorithm", "combined", "local (trust.)", "global (pair ρ)", "triplet",
              "fit s", "peak GB"]
    rows = [header, ["---"] * len(header)]
    for p in sorted(points, key=lambda p: -combined[p.algorithm]):
        rows.append([
            p.algorithm, f"{combined[p.algorithm]:.2f}", with_sd(p, "local", "{:.4f}"),
            with_sd(p, "global", "{:.3f}"), with_sd(p, "triplet_acc", "{:.3f}"),
            f"{p.mean['fit_s']:.1f}", f"{p.mean['peak_rss_mb'] / MB_PER_GB:.1f}",
        ])
    return "\n".join("| " + " | ".join(cells) + " |" for cells in rows)


def write_scaling_report(config: Config) -> Path:
    results_dir = config.run.results_dir
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    specs = config.run.datasets
    points = aggregate(joined_records(results_dir, specs, config.run.algorithms))
    if not points:
        raise ValueError(f"no scored runs for {specs} in {results_dir}")
    dataset = datasets.parse_spec(specs[0])[0]
    chart = figures / f"{dataset}-quality-map.png"
    plot_quality_map(points, dataset, chart)

    metrics = config.metrics
    sections = [
        f"# Projection quality vs N — {dataset}\n",
        (f"{config.run.n_threads} threads · seeds {list(config.run.seeds)} · mean ± sd over seeds. "
         "**combined** = mean of local and global after min-max scaling each across the methods "
         "at that N (relative: compare within one N only). "
         f"Local scores use {metrics.anchor_rows:,} anchor points above N = "
         f"{metrics.anchors_above_n:,}, every point otherwise.\n"),
        f"![Projection quality]({chart.relative_to(results_dir)})\n",
    ]
    for n in sorted({p.n for p in points}):
        sections += [f"## N = {n:,}\n", quality_table([p for p in points if p.n == n]) + "\n"]
    report = results_dir / "SCALING.md"
    report.write_text("\n".join(sections))
    print(f"wrote {report}")
    return report
