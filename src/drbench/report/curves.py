"""Curve report: quality-vs-time charts, every curve point, and the best score reachable within
fixed wall-time limits (the fair 'same compute' comparison)."""

from pathlib import Path

from drbench import algorithms
from drbench.config import Config
from drbench.report.plots import plot_curves
from drbench.runner.records import latest_by_key, metrics_path, read_jsonl, runs_path

TIME_LIMITS_S = (5, 15, 30, 60, 120, 300)
MISSING = "–"


def curve_points(results_dir: Path, datasets: tuple[str, ...]) -> list[dict]:
    """Join successful runs with their scores; one dict per (dataset, algorithm, budget)."""
    runs = latest_by_key(read_jsonl(runs_path(results_dir)))
    scores = latest_by_key(read_jsonl(metrics_path(results_dir)))
    points = []
    for key, run in sorted(runs.items()):
        if run["status"] != "ok" or key not in scores or key.dataset not in datasets:
            continue
        algorithm = algorithms.get(key.algorithm)
        points.append(run | scores[key] | {"family": algorithm.family, "library": algorithm.library})
    return points


def points_table(points: list[dict]) -> str:
    header = ["algorithm", "budget", "epochs", "wall s", "CPU s", "cores used", "peak MB",
              "local", "global", "T@15", "C@15"]
    rows = [header, ["---"] * len(header)]
    for p in sorted(points, key=lambda p: (p["algorithm"], p["budget"])):
        rows.append([
            p["algorithm"], f"×{p['budget']:g}", str(p.get("epochs") or MISSING),
            f"{p['fit_s']:.1f}", f"{p['cpu_s']:.0f}", f"{p['cpu_s'] / p['fit_s']:.1f}",
            f"{p['peak_rss_mb']:.0f}", f"{p['local']:.4f}", f"{p['global']:.4f}",
            f"{p['trust_15']:.4f}", f"{p['continuity_15']:.4f}",
        ])
    return "\n".join("| " + " | ".join(cells) + " |" for cells in rows)


def best_within(points: list[dict], algorithm: str, limit_s: float) -> dict | None:
    eligible = [p for p in points if p["algorithm"] == algorithm and p["fit_s"] <= limit_s]
    return max(eligible, key=lambda p: p["local"], default=None)


def time_limit_table(points: list[dict]) -> str:
    """Best local score each algorithm reaches within each wall-time limit (budget in brackets)."""
    names = sorted({p["algorithm"] for p in points})
    header = ["algorithm", *[f"≤ {limit} s" for limit in TIME_LIMITS_S]]
    rows = [header, ["---"] * len(header)]
    for name in names:
        cells = [name]
        for limit in TIME_LIMITS_S:
            best = best_within(points, name, limit)
            cells.append(f"{best['local']:.4f} (×{best['budget']:g})" if best else MISSING)
        rows.append(cells)
    winners = ["**best**"]
    for limit in TIME_LIMITS_S:
        eligible = [p for p in points if p["fit_s"] <= limit]
        best = max(eligible, key=lambda p: p["local"], default=None)
        winners.append(f"**{best['algorithm']}** {best['local']:.4f}" if best else MISSING)
    rows.append(winners)
    return "\n".join("| " + " | ".join(cells) + " |" for cells in rows)


def write_curves_report(config: Config) -> Path:
    results_dir = config.run.results_dir
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    budgets = ", ".join(f"×{b:g}" for b in config.curves.budgets)
    method = (
        f"Each tunable algorithm reruns with its default epoch count scaled by {budgets} "
        f"(extended to ×{config.curves.extend_budget:g} if still improving by more than "
        f"{config.curves.extend_min_gain} local score). Seed {config.curves.seed}, "
        f"{config.run.n_threads} threads. Every point is a separate full run (kNN included). "
        f"Local scores are estimated on {config.curves.anchor_rows} fixed anchor points, with "
        "exact ranks over all N. 'cores used' = CPU s / wall s.\n"
    )
    sections = ["# Quality vs time (epoch-budget curves)\n", method]
    for spec in config.run.datasets:
        points = curve_points(results_dir, (spec,))
        if not points:
            continue
        chart = figures / f"{spec}-curves.png"
        plot_curves(points, spec, chart)
        sections += [
            f"## {spec}\n", f"![Curves]({chart.relative_to(results_dir)})\n",
            "### Best local score within a wall-time limit\n", time_limit_table(points) + "\n",
            "### All curve points\n", points_table(points) + "\n",
        ]
    report = results_dir / "CURVES.md"
    report.write_text("\n".join(sections))
    print(f"wrote {report}")
    return report
