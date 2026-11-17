"""Dataset registry.

A dataset spec is `name` or `name@N`. `@N` takes a fixed random subsample of N rows, so
quick runs and the full run share one code path and keep separate caches.
Each dataset module provides `prepare(target_dir)`, which writes `X.npy` (float32, rows = samples).
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drbench.datasets import dbpedia, derived

SUBSAMPLE_SEED = 0
LOW_DIM_COMPONENTS = 32
SPEC_SEPARATOR = "@"


@dataclass(frozen=True)
class Dataset:
    name: str
    description: str
    prepare: Callable[[Path], None]


REGISTRY: dict[str, Dataset] = {
    "dbpedia-512-100k": Dataset(
        name="dbpedia-512-100k",
        description="100k DBpedia entities, OpenAI text-embedding-3-small (512d), L2-normalized",
        prepare=dbpedia.prepare,
    ),
    "dbpedia-pca32-100k": Dataset(
        name="dbpedia-pca32-100k",
        description=f"dbpedia-512-100k projected to {LOW_DIM_COMPONENTS} dimensions with PCA",
        prepare=derived.pca_projection(dbpedia.prepare, "dbpedia-512-100k", LOW_DIM_COMPONENTS),
    ),
}


def parse_spec(spec: str) -> tuple[str, int | None]:
    name, _, size = spec.partition(SPEC_SEPARATOR)
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(REGISTRY)}")
    return name, int(size) if size else None


def with_subsample(spec: str, n_rows: int | None) -> str:
    name, _ = parse_spec(spec)
    return f"{name}{SPEC_SEPARATOR}{n_rows}" if n_rows else name


def cache_dir(spec: str, data_dir: Path) -> Path:
    """Directory for derived per-spec artefacts (reference neighbours, clusters)."""
    return data_dir / spec


def subsample(X: np.ndarray, n_rows: int) -> np.ndarray:
    rows = np.sort(np.random.default_rng(SUBSAMPLE_SEED).choice(len(X), n_rows, replace=False))
    return np.ascontiguousarray(X[rows], dtype=np.float32)


def prepare(spec: str, data_dir: Path) -> Path:
    """Materialise `X.npy` for a spec. Subsamples get their own file, so workers never touch
    (and count in their RSS) the full dataset."""
    name, n_rows = parse_spec(spec)
    full_path = data_dir / name / "X.npy"
    if not full_path.exists():
        full_path.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY[name].prepare(full_path.parent)
    spec_path = cache_dir(spec, data_dir) / "X.npy"
    if n_rows is not None and not spec_path.exists():
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(spec_path, subsample(np.load(full_path), n_rows))
    return spec_path


def load(spec: str, data_dir: Path) -> np.ndarray:
    path = cache_dir(spec, data_dir) / "X.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `drbench prep` first")
    return np.ascontiguousarray(np.load(path), dtype=np.float32)
