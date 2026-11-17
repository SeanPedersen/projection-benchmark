"""UMAP adapters (umap-learn reference, manifolds-rs) plus manifolds-rs PaCMAP.

Default epochs:
- umap-learn: 200 for N > 10k, 500 below, in both pipelines. The kNN backend is pinned per variant.
- manifolds-rs UMAP: 500 with the Adam optimiser.
- manifolds-rs PaCMAP: 450.

The `-rinnd-fast` / `-rinnd-default` variants take the shared RINND graph (`knn_graph.py`, fast
or default schedule) as a precomputed kNN graph: k=15 for UMAP (umap-learn counts the point itself), `n_near + 50` = 60 for PaCMAP, which
picks its near pairs from that wider search.
"""

from functools import partial

import numpy as np

from drbench.algorithms import knn_graph
from drbench.algorithms.params import MIN_DIST, N_NEIGHBORS, scaled_epochs

UMAP_LEARN_SMALL_N = 10_000
UMAP_LEARN_EPOCHS_SMALL = 500
UMAP_LEARN_EPOCHS_LARGE = 200
MANIFOLDS_UMAP_EPOCHS = 500
MANIFOLDS_PACMAP_EPOCHS = 450
PACMAP_N_NEAR = 10
PACMAP_SEARCH_WIDTH = PACMAP_N_NEAR + 50


def _umap_learn(
    X: np.ndarray,
    seed: int,
    n_threads: int,
    budget: float,
    *,
    legacy: bool,
    nn_backend: str,
    graph: str | None = None,
) -> np.ndarray:
    import umap

    default_epochs = UMAP_LEARN_EPOCHS_SMALL if len(X) <= UMAP_LEARN_SMALL_N else UMAP_LEARN_EPOCHS_LARGE
    # A fixed random_state forces umap-learn single-threaded, so seed the global RNG instead.
    np.random.seed(seed)
    model = umap.UMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        n_epochs=scaled_epochs(default_epochs, budget),
        n_jobs=n_threads,
        random_state=None,
        compatibility_layout=legacy,
        nearest_neighbor_backend=nn_backend,
        precomputed_knn=(
            knn_graph.with_self(*knn_graph.rinnd_graph(X, N_NEIGHBORS, n_threads, graph),
                                N_NEIGHBORS)
            if graph is not None else (None, None, None)
        ),
    )
    return model.fit_transform(X)


# umap-learn >= 0.6 prefers RINND when installed and silently falls back to PyNNDescent otherwise.
# Pinning the backend keeps every variant reproducible across environments.
umap_learn = partial(_umap_learn, legacy=False, nn_backend="pynndescent")
umap_learn_rinnd = partial(_umap_learn, legacy=False, nn_backend="rinnd")
umap_learn_rinnd_fast = partial(_umap_learn, legacy=False, nn_backend="rinnd", graph=knn_graph.FAST)
umap_learn_rinnd_default = partial(_umap_learn, legacy=False, nn_backend="rinnd",
                                   graph=knn_graph.DEFAULT)
# umap-learn 0.5.x pipeline: spectral initialization + SGD layout (always PyNNDescent).
umap_learn_legacy = partial(_umap_learn, legacy=True, nn_backend="auto")


def shared_graph_kwargs(X: np.ndarray, k: int, n_threads: int, graph: str) -> dict:
    indices, distances = knn_graph.rinnd_graph(X, k, n_threads, graph)
    return {"knn_indices": indices, "knn_distances": distances}


def _manifolds_umap(
    X: np.ndarray, seed: int, n_threads: int, budget: float, *, graph: str | None = None
) -> np.ndarray:
    import manifolds_rs as mf

    mf.set_num_threads(n_threads)
    model = mf.UMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        n_epochs=scaled_epochs(MANIFOLDS_UMAP_EPOCHS, budget),
        seed=seed,
    )
    knn = shared_graph_kwargs(X, N_NEIGHBORS, n_threads, graph) if graph is not None else {}
    return model.fit_transform(X, **knn)


def _manifolds_pacmap(
    X: np.ndarray, seed: int, n_threads: int, budget: float, *, graph: str | None = None
) -> np.ndarray:
    import manifolds_rs as mf

    mf.set_num_threads(n_threads)
    model = mf.PaCMAP(n_components=2, n_near=PACMAP_N_NEAR,
                      n_epochs=scaled_epochs(MANIFOLDS_PACMAP_EPOCHS, budget), seed=seed)
    knn = (shared_graph_kwargs(X, PACMAP_SEARCH_WIDTH, n_threads, graph)
           if graph is not None else {})
    return model.fit_transform(X, **knn)


manifolds_umap = _manifolds_umap
manifolds_umap_rinnd_fast = partial(_manifolds_umap, graph=knn_graph.FAST)
manifolds_umap_rinnd_default = partial(_manifolds_umap, graph=knn_graph.DEFAULT)
manifolds_pacmap = _manifolds_pacmap
manifolds_pacmap_rinnd_fast = partial(_manifolds_pacmap, graph=knn_graph.FAST)
manifolds_pacmap_rinnd_default = partial(_manifolds_pacmap, graph=knn_graph.DEFAULT)
