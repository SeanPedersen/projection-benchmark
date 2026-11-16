"""Quality-vs-time curves: rerun each algorithm across a grid of iteration budgets.

Epoch counts cannot be compared across algorithms, so the budget (a multiplier on each library's
default epochs) is only the knob; the curves are plotted against wall time and CPU time. Every
point is a separate full run, so fixed costs (kNN, affinities) appear as each curve's offset
on the time axis. Algorithms still improving at the largest budget get one extra doubling.
"""

from dataclasses import replace
from pathlib import Path

from drbench import algorithms
from drbench.config import Config
from drbench.runner.evaluate import evaluate_pending
from drbench.runner.orchestrator import run_sweep, shuffled
from drbench.runner.records import RunKey, latest_by_key, metrics_path, read_jsonl


def curves_config(config: Config, results_dir: Path | None = None) -> Config:
    """The run/metrics settings that a curve sweep uses (one seed, anchor-sampled scores)."""
    curves = config.curves
    return replace(
        config,
        run=replace(
            config.run,
            algorithms=curves.algorithms,
            seeds=(curves.seed,),
            results_dir=results_dir or curves.results_dir,
        ),
        metrics=replace(config.metrics, anchor_rows=curves.anchor_rows),
    )


def budgets_for(algorithm: str, budgets: tuple[float, ...]) -> tuple[float, ...]:
    return budgets if algorithms.get(algorithm).tunable else (1.0,)


def curve_plan(config: Config) -> list[RunKey]:
    run, curves = config.run, config.curves
    return shuffled([
        RunKey(dataset, algorithm, curves.seed, budget)
        for dataset in run.datasets
        for algorithm in run.algorithms
        for budget in budgets_for(algorithm, curves.budgets)
    ])


def still_improving(config: Config) -> list[RunKey]:
    """Extension runs for curves whose local score still rises between the two largest budgets."""
    curves = config.curves
    scores = latest_by_key(read_jsonl(metrics_path(config.run.results_dir)))
    previous_budget, last_budget = sorted(curves.budgets)[-2:]
    extensions = []
    for dataset in config.run.datasets:
        for algorithm in config.run.algorithms:
            if not algorithms.get(algorithm).tunable:
                continue
            last = scores.get(RunKey(dataset, algorithm, curves.seed, last_budget))
            previous = scores.get(RunKey(dataset, algorithm, curves.seed, previous_budget))
            if last and previous and last["local"] - previous["local"] > curves.extend_min_gain:
                extensions.append(RunKey(dataset, algorithm, curves.seed, curves.extend_budget))
    return extensions


def run_curves(config: Config, retry_failed: bool = False) -> None:
    run_sweep(config, curve_plan(config), retry_failed)
    evaluate_pending(config)
    extensions = still_improving(config)
    if extensions:
        print(f"extending: {[key.algorithm for key in extensions]} to ×{config.curves.extend_budget:g}")
        run_sweep(config, extensions, retry_failed)
        evaluate_pending(config)
