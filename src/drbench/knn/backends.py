"""kNN backend registry. Each builder maps `(X, k, n_threads) -> (n, k)` neighbour indices
without the point itself. Imports are lazy, so only the benchmarked library is loaded."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

BACKEND_SEED = 0

KnnBuilder = Callable[[np.ndarray, int, int], np.ndarray]


@dataclass(frozen=True)
class KnnBackend:
    name: str
    library: str
    package: str
    exact: bool
    build: KnnBuilder


def _manifolds(X: np.ndarray, k: int, n_threads: int, *, ann: str) -> np.ndarray:
    import manifolds_rs as mf

    mf.set_num_threads(n_threads)
    return mf.knn_graph(X, k=k, ann=ann, seed=BACKEND_SEED)[0]


def _pynndescent(X: np.ndarray, k: int, n_threads: int) -> np.ndarray:
    from pynndescent import NNDescent

    # The graph includes each point as its own nearest neighbour, so ask for one extra.
    return NNDescent(X, n_neighbors=k + 1, n_jobs=n_threads).neighbor_graph[0][:, 1:]


def _rinnd(X: np.ndarray, k: int, n_threads: int) -> np.ndarray:
    from rinnd import RINND

    return RINND(X, n_neighbors=k, n_jobs=n_threads, graph_only=True).neighbor_graph[0]


def _opentsne_annoy(X: np.ndarray, k: int, n_threads: int) -> np.ndarray:
    from openTSNE.nearest_neighbors import Annoy

    return Annoy(X, k, "euclidean", n_jobs=n_threads, random_state=BACKEND_SEED).build()[0]


def _manifolds_backend(ann: str, *, exact: bool = False) -> KnnBackend:
    return KnnBackend(
        f"mrs-{ann}", "manifolds-rs", "manifolds-rs", exact, partial(_manifolds, ann=ann)
    )


REGISTRY: dict[str, KnnBackend] = {
    backend.name: backend
    for backend in (
        KnnBackend("rinnd", "RINND", "rinnd", False, _rinnd),
        KnnBackend("pynndescent", "PyNNDescent", "pynndescent", False, _pynndescent),
        KnnBackend("opentsne-annoy", "openTSNE (Annoy)", "openTSNE", False, _opentsne_annoy),
        _manifolds_backend("nndescent"),
        _manifolds_backend("ivf"),
        _manifolds_backend("hnsw"),
        _manifolds_backend("annoy"),
        _manifolds_backend("balltree"),
        _manifolds_backend("exhaustive", exact=True),
        _manifolds_backend("kmknn", exact=True),
    )
}


def get(name: str) -> KnnBackend:
    if name not in REGISTRY:
        raise KeyError(f"unknown kNN backend {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]
