"""Write `summary.csv`, figures and `REPORT.md` into the results directory."""

from datetime import date
from pathlib import Path

import numpy as np

from drbench import datasets
from drbench.config import Config
from drbench.report.plots import plot_pareto, plot_thumbnails
from drbench.report.tables import Summary, summarize, write_csv
from drbench.runner.records import RunKey

THUMBNAIL_SEED = 0


MISSING = "–"


def format_score(scores: dict[str, float], name: str) -> str:
    return f"{scores[name]:.4f}" if name in scores else MISSING


def table_row(s: Summary, k_local: tuple[int, ...]) -> list[str]:
    mean, sd = s.score_mean, s.score_sd
    recall = next((name for name in mean if name.startswith("recall_")), "recall")
    def with_sd(name: str) -> str:
        return f"{mean[name]:.4f} ± {sd[name]:.4f}" if name in mean else MISSING
    return [
        s.algorithm, s.library,
        f"{s.cost['fit_s']:.1f}" if s.cost else MISSING,
        f"{s.cost['peak_rss_mb']:.0f}" if s.cost else MISSING,
        with_sd("local"), with_sd("global"),
        *[format_score(mean, f"trust_{k}") for k in k_local],
        *[format_score(mean, f"continuity_{k}") for k in k_local],
        format_score(mean, recall), format_score(mean, "pair_rho"), f"{s.n_ok}/{s.n_failed}",
    ]


def markdown_table(summaries: list[Summary], k_local: tuple[int, ...]) -> str:
    columns = ["algorithm", "library", "fit s", "peak MB", "local", "global",
               *[f"T@{k}" for k in k_local], *[f"C@{k}" for k in k_local],
               "recall", "pair ρ", "ok/failed"]
    rows = [columns, ["---"] * len(columns), *[table_row(s, k_local) for s in summaries]]
    return "\n".join("| " + " | ".join(cells) + " |" for cells in rows)


def report_header(config: Config) -> str:
    run, metrics = config.run, config.metrics
    return (
        "# Dimensionality reduction benchmark\n\n"
        f"Generated {date.today().isoformat()} · {run.n_threads} threads · seeds {list(run.seeds)}\n\n"
        f"**local** = mean trustworthiness over k ∈ {list(metrics.k_local)}; "
        f"**global** = mean Spearman ρ of k-means centroid distances over k ∈ {list(metrics.k_clusters)}. "
        "T = trustworthiness, C = continuity. Cost = median over seeds; scores = mean ± sd.\n"
    )


def thumbnails_for(spec: str, summaries: list[Summary], config: Config, path: Path) -> bool:
    reference = datasets.cache_dir(spec, config.run.data_dir) / "reference.npz"
    k_color = min(config.metrics.k_clusters)
    embeddings = {}
    for s in summaries:
        embedding = RunKey(spec, s.algorithm, THUMBNAIL_SEED).embedding_path(config.run.results_dir)
        if embedding.exists():
            embeddings[s.algorithm] = np.load(embedding)
    if not embeddings or not reference.exists():
        return False
    labels = np.load(reference)[f"k{k_color}_labels"]
    title = f"{spec} — seed {THUMBNAIL_SEED}, coloured by k-means (k={k_color})"
    plot_thumbnails(embeddings, labels, title, path)
    return True


def write_report(config: Config) -> Path:
    results_dir = config.run.results_dir
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summaries = summarize(results_dir, config.run.datasets)
    write_csv(summaries, results_dir / "summary.csv")

    sections = [report_header(config)]
    for spec in config.run.datasets:
        rows = [s for s in summaries if s.dataset == spec]
        if not rows:
            continue
        pareto = figures / f"{spec}-pareto.png"
        plot_pareto(rows, spec, pareto)
        sections += [f"## {spec}\n", markdown_table(rows, config.metrics.k_local) + "\n",
                     f"![Pareto]({pareto.relative_to(results_dir)})\n"]
        thumbs = figures / f"{spec}-embeddings.png"
        if thumbnails_for(spec, rows, config, thumbs):
            sections.append(f"![Embeddings]({thumbs.relative_to(results_dir)})\n")
    report = results_dir / "REPORT.md"
    report.write_text("\n".join(sections))
    print(f"wrote {report}")
    return report
