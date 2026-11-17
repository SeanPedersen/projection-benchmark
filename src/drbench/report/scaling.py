"""Scaling report: one dataset at several subsample sizes (`name@N`), every algorithm, all seeds.

Writes `SCALING.md` (algorithm × N table, mean ± sd over seeds) and three charts into the results
directory: time vs N (fit and kNN graph build), peak memory vs N, quality vs N.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drbench import algorithms, datasets
from drbench.config import Config
from drbench.report.plots import plot_scaling_memory, plot_scaling_quality, plot_scaling_time
from drbench.runner.evaluate import current_scores
from drbench.runner.records import latest_by_key, read_jsonl, runs_path

COLUMNS = (  # (record field, header, format)
    ("fit_s", "fit s", "{:.1f}"),
    ("knn_s", "kNN s", "{:.1f}"),
    ("layout_s", "layout s", "{:.1f}"),
    ("peak_rss_mb", "peak MB", "{:.0f}"),
    ("local", "local", "{:.4f}"),
    ("global", "global", "{:.3f}"),
    ("triplet_acc", "triplet", "{:.3f}"),
)
MISSING = "–"


@dataclass(frozen=True)
class Point:
    """Mean and sd over seeds of one algorithm at one N."""

    algorithm: str
    n: int
    n_seeds: int
    mean: dict[str, float]
    sd: dict[str, float]

    @property
    def family(self) -> str:
        return algorithms.get(self.algorithm).family

    @property
    def library(self) -> str:
        return algorithms.get(self.algorithm).library


def joined_records(
    results_dir: Path, specs: tuple[str, ...], names: tuple[str, ...]
) -> dict[tuple[str, int], list[dict]]:
    """Successful, scored runs of `names` grouped by (algorithm, N), with derived `layout_s`."""
    runs = latest_by_key(read_jsonl(runs_path(results_dir)))
    scores = current_scores(results_dir)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for key, run in runs.items():
        if (key.dataset not in specs or key.algorithm not in names or run["status"] != "ok"
                or key not in scores):
            continue
        knn_s = run.get("knn_s") or 0.0
        grouped[(key.algorithm, run["n"])].append(
            run | scores[key] | {"knn_s": knn_s, "layout_s": run["fit_s"] - knn_s}
        )
    return grouped


def aggregate(grouped: dict[tuple[str, int], list[dict]]) -> list[Point]:
    points = []
    for (algorithm, n), records in sorted(grouped.items()):
        values = {field: np.array([r[field] for r in records]) for field, _, _ in COLUMNS}
        points.append(Point(
            algorithm, n, len(records),
            mean={f: float(v.mean()) for f, v in values.items()},
            sd={f: float(v.std(ddof=1)) if len(v) > 1 else 0.0 for f, v in values.items()},
        ))
    return points


def algorithm_order(points: list[Point]) -> list[str]:
    """Fastest first at the largest N."""
    largest = max(p.n for p in points)
    at_largest = {p.algorithm: p.mean["fit_s"] for p in points if p.n == largest}
    names = sorted({p.algorithm for p in points})
    return sorted(names, key=lambda name: at_largest.get(name, np.inf))


def cell(point: Point | None, field: str, fmt: str) -> str:
    if point is None:
        return MISSING
    mean, sd = point.mean[field], point.sd[field]
    return f"{fmt.format(mean)} ± {fmt.format(sd)}" if point.n_seeds > 1 else fmt.format(mean)


def markdown_table(points: list[Point]) -> str:
    sizes = sorted({p.n for p in points})
    by_key = {(p.algorithm, p.n): p for p in points}
    header = ["algorithm", "N", *[title for _, title, _ in COLUMNS], "seeds"]
    rows = [header, ["---"] * len(header)]
    for name in algorithm_order(points):
        for n in sizes:
            point = by_key.get((name, n))
            rows.append([name, f"{n:,}", *[cell(point, f, fmt) for f, _, fmt in COLUMNS],
                         str(point.n_seeds) if point else "0"])
    return "\n".join("| " + " | ".join(cells) + " |" for cells in rows)


def write_scaling_report(config: Config) -> Path:
    results_dir = config.run.results_dir
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    specs = config.run.datasets
    points = aggregate(joined_records(results_dir, specs, config.run.algorithms))
    if not points:
        raise ValueError(f"no scored runs for {specs} in {results_dir}")
    names = algorithm_order(points)
    dataset = datasets.parse_spec(specs[0])[0]
    charts = {
        "time": figures / f"{dataset}-scaling-time.png",
        "memory": figures / f"{dataset}-scaling-memory.png",
        "quality": figures / f"{dataset}-scaling-quality.png",
    }
    plot_scaling_time(points, names, dataset, charts["time"])
    plot_scaling_memory(points, names, dataset, charts["memory"])
    plot_scaling_quality(points, names, dataset, charts["quality"])

    metrics = config.metrics
    sections = [
        f"# Scaling with N — {dataset}\n",
        (f"{config.run.n_threads} threads · seeds {list(config.run.seeds)} · mean ± sd over seeds. "
        "kNN s = build time of the shared RINND graph (charged to every run that uses it); "
        "layout s = fit s − kNN s. "
        f"Local scores use {metrics.anchor_rows:,} anchor points above N = "
        f"{metrics.anchors_above_n:,}, every point otherwise.\n"),
        markdown_table(points) + "\n",
        *[f"![{title}]({path.relative_to(results_dir)})\n" for title, path in charts.items()],
    ]
    report = results_dir / "SCALING.md"
    report.write_text("\n".join(sections))
    print(f"wrote {report}")
    return report
