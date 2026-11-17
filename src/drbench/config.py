"""Load `benchmark.toml` into frozen dataclasses."""

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    datasets: tuple[str, ...]
    algorithms: tuple[str, ...]
    seeds: tuple[int, ...]
    n_threads: int
    timeout_s: float
    cooldown_s: float
    warmup_rows: int
    data_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class MetricsConfig:
    k_local: tuple[int, ...]
    color_clusters: int
    recall_k: int
    n_random_pairs: int
    n_triplets: int
    chunk_rows: int
    # Local scores use `anchor_rows` fixed anchor points once N exceeds `anchors_above_n`.
    anchor_rows: int | None = None
    anchors_above_n: int = 0

    @property
    def k_max(self) -> int:
        return max(*self.k_local, self.recall_k)


@dataclass(frozen=True)
class CurvesConfig:
    algorithms: tuple[str, ...]
    budgets: tuple[float, ...]
    seed: int
    extend_min_gain: float
    extend_budget: float
    anchor_rows: int
    results_dir: Path


@dataclass(frozen=True)
class KnnConfig:
    datasets: tuple[str, ...]
    backends: tuple[str, ...]
    ks: tuple[int, ...]
    repeats: int
    warmup_rows: int
    results_dir: Path


@dataclass(frozen=True)
class Config:
    run: RunConfig
    metrics: MetricsConfig
    curves: CurvesConfig
    knn: KnnConfig

    def with_knn(self, **overrides) -> "Config":
        present = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, knn=replace(self.knn, **present))

    def with_run(self, **overrides) -> "Config":
        present = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, run=replace(self.run, **present))


def load_config(path: Path) -> Config:
    raw = tomllib.loads(path.read_text())
    run, metrics, curves, knn = raw["run"], raw["metrics"], raw["curves"], raw["knn"]
    return Config(
        run=RunConfig(
            datasets=tuple(run["datasets"]),
            algorithms=tuple(run["algorithms"]),
            seeds=tuple(run["seeds"]),
            n_threads=int(run["n_threads"]),
            timeout_s=float(run["timeout_s"]),
            cooldown_s=float(run["cooldown_s"]),
            warmup_rows=int(run["warmup_rows"]),
            data_dir=Path(run["data_dir"]),
            results_dir=Path(run["results_dir"]),
        ),
        metrics=MetricsConfig(
            k_local=tuple(metrics["k_local"]),
            color_clusters=int(metrics["color_clusters"]),
            recall_k=int(metrics["recall_k"]),
            n_random_pairs=int(metrics["n_random_pairs"]),
            n_triplets=int(metrics["n_triplets"]),
            chunk_rows=int(metrics["chunk_rows"]),
            anchor_rows=metrics.get("anchor_rows"),
            anchors_above_n=int(metrics.get("anchors_above_n", 0)),
        ),
        curves=CurvesConfig(
            algorithms=tuple(curves["algorithms"]),
            budgets=tuple(float(b) for b in curves["budgets"]),
            seed=int(curves["seed"]),
            extend_min_gain=float(curves["extend_min_gain"]),
            extend_budget=float(curves["extend_budget"]),
            anchor_rows=int(curves["anchor_rows"]),
            results_dir=Path(curves["results_dir"]),
        ),
        knn=KnnConfig(
            datasets=tuple(knn["datasets"]),
            backends=tuple(knn["backends"]),
            ks=tuple(int(k) for k in knn["ks"]),
            repeats=int(knn["repeats"]),
            warmup_rows=int(knn["warmup_rows"]),
            results_dir=Path(knn["results_dir"]),
        ),
    )
