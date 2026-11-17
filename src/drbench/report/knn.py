"""Render `KNN.md`: median wall time and recall per (dataset, backend, k), with the spread over
repeats."""

import statistics
from collections import defaultdict

from drbench.config import Config
from drbench.runner.records import read_jsonl

HEADER = "| backend | k | time median (min-max) | recall median (min-max) | repeats |\n|---|---|---|---|---|\n"


def span(values: list[float], digits: int) -> str:
    return f"{statistics.median(values):.{digits}f} ({min(values):.{digits}f}-{max(values):.{digits}f})"


def group_ok(records: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        if record["status"] == "ok":
            groups[(record["dataset"], record["backend"], record["k"])].append(record)
    return groups


def render(records: list[dict]) -> str:
    groups = group_ok(records)
    sections = []
    for dataset in sorted({key[0] for key in groups}):
        sample = next(r for key, rs in groups.items() if key[0] == dataset for r in rs)
        rows = sorted(
            (key for key in groups if key[0] == dataset),
            key=lambda key: (key[2], statistics.median(r["time_s"] for r in groups[key])),
        )
        lines = [
            f"| {backend} | {k} | {span([r['time_s'] for r in groups[key]], 1)} s "
            f"| {span([r['recall'] for r in groups[key]], 4)} | {len(groups[key])} |"
            for key in rows for _, backend, k in [key]
        ]
        title = f"## {dataset} (n={sample['n']}, d={sample['dim']}, {sample['n_threads']} threads)"
        sections.append(f"{title}\n\n{HEADER}" + "\n".join(lines))
    return "# kNN backends\n\nRecall is measured against the exact graph.\n\n" + "\n\n".join(sections) + "\n"


def write_knn_report(config: Config) -> None:
    results_dir = config.knn.results_dir
    records = read_jsonl(results_dir / "knn.jsonl")
    (results_dir / "KNN.md").write_text(render(records))
    print(f"wrote {results_dir / 'KNN.md'}")
