"""UMAP adapters (umap-learn reference, manifolds-rs) plus manifolds-rs PaCMAP.

Default epochs:
- umap-learn: 200 for N > 10k, 500 below, in both pipelines.
- manifolds-rs UMAP: 500 with the Adam optimiser.
- manifolds-rs PaCMAP: 450.
"""

import numpy as np

from drbench.algorithms.params import MIN_DIST, N_NEIGHBORS, scaled_epochs

UMAP_LEARN_SMALL_N = 10_000
UMAP_LEARN_EPOCHS_SMALL = 500
UMAP_LEARN_EPOCHS_LARGE = 200
MANIFOLDS_UMAP_EPOCHS = 500
MANIFOLDS_PACMAP_EPOCHS = 450


def _umap_learn(X: np.ndarray, seed: int, n_threads: int, budget: float, legacy: bool) -> np.ndarray:
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
    )
    return model.fit_transform(X)


def umap_learn(X: np.ndarray, seed: int, n_threads: int, budget: float) -> np.ndarray:
    """umap-learn >= 0.6 pipeline: recursive initialization + Adam layout."""
    return _umap_learn(X, seed, n_threads, budget, legacy=False)


def umap_learn_legacy(X: np.ndarray, seed: int, n_threads: int, budget: float) -> np.ndarray:
    """umap-learn 0.5.x pipeline: spectral initialization + SGD layout."""
    return _umap_learn(X, seed, n_threads, budget, legacy=True)


def manifolds_umap(X: np.ndarray, seed: int, n_threads: int, budget: float) -> np.ndarray:
    import manifolds_rs as mf

    mf.set_num_threads(n_threads)
    model = mf.UMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        n_epochs=scaled_epochs(MANIFOLDS_UMAP_EPOCHS, budget),
        seed=seed,
    )
    return model.fit_transform(X)


def manifolds_pacmap(X: np.ndarray, seed: int, n_threads: int, budget: float) -> np.ndarray:
    import manifolds_rs as mf

    mf.set_num_threads(n_threads)
    model = mf.PaCMAP(n_components=2, n_epochs=scaled_epochs(MANIFOLDS_PACMAP_EPOCHS, budget), seed=seed)
    return model.fit_transform(X)
