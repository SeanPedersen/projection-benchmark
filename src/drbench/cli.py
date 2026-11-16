"""Command-line entry point: `drbench {prep,run,evaluate,report,all}`."""

import argparse
from pathlib import Path

from drbench import datasets
from drbench.config import Config, load_config

DEFAULT_CONFIG = Path("benchmark.toml")


def csv_list(cast=str):
    return lambda text: tuple(cast(item) for item in text.split(",") if item)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    common.add_argument("--datasets", type=csv_list(), help="comma-separated dataset names")
    common.add_argument("--algos", type=csv_list(), help="comma-separated algorithm names")
    common.add_argument("--seeds", type=csv_list(int), help="comma-separated seeds")
    common.add_argument("--subsample", type=int, help="use a fixed random subset of N rows")
    common.add_argument("--threads", type=int)
    common.add_argument("--timeout", type=float, help="per-run timeout in seconds")
    common.add_argument("--results-dir", type=Path)

    parser = argparse.ArgumentParser(prog="drbench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prep", parents=[common], help="download and convert datasets")
    run = commands.add_parser("run", parents=[common], help="run the benchmark sweep")
    run.add_argument("--retry-failed", action="store_true")
    commands.add_parser("evaluate", parents=[common], help="score finished runs")
    commands.add_parser("report", parents=[common], help="write summary.csv, plots, REPORT.md")
    everything = commands.add_parser("all", parents=[common], help="prep, run, evaluate, report")
    everything.add_argument("--retry-failed", action="store_true")
    curves = commands.add_parser(
        "curves", parents=[common], help="quality vs time over a grid of epoch budgets"
    )
    curves.add_argument("--retry-failed", action="store_true")
    return parser


def resolve_config(args: argparse.Namespace) -> Config:
    config = load_config(args.config)
    specs = args.datasets or config.run.datasets
    return config.with_run(
        datasets=tuple(datasets.with_subsample(spec, args.subsample) for spec in specs),
        algorithms=args.algos,
        seeds=args.seeds,
        n_threads=args.threads,
        timeout_s=args.timeout,
        results_dir=args.results_dir,
    )


def prep(config: Config) -> None:
    for spec in config.run.datasets:
        print(f"{spec}: {datasets.prepare(spec, config.run.data_dir)}")


def run_curves_command(args: argparse.Namespace, config: Config) -> None:
    from drbench.report.curves import write_curves_report
    from drbench.runner.curves import curves_config, run_curves

    curve_config = curves_config(config, results_dir=args.results_dir)
    if args.algos:
        curve_config = curve_config.with_run(algorithms=args.algos)
    prep(curve_config)
    run_curves(curve_config, retry_failed=args.retry_failed)
    write_curves_report(curve_config)


def main() -> None:
    args = build_parser().parse_args()
    config = resolve_config(args)
    if args.command == "curves":
        run_curves_command(args, config)
        return
    # Heavy modules are imported per command so `drbench prep` stays light.
    if args.command in ("prep", "run", "all"):
        prep(config)
    if args.command in ("run", "all"):
        from drbench.runner.orchestrator import default_plan, run_sweep
        run_sweep(config, default_plan(config), retry_failed=args.retry_failed)
    if args.command in ("evaluate", "all"):
        from drbench.runner.evaluate import evaluate_pending
        evaluate_pending(config)
    if args.command in ("report", "all"):
        from drbench.report import write_report
        write_report(config)


if __name__ == "__main__":
    main()
