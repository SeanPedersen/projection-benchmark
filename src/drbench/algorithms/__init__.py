"""Algorithm registry.

Each adapter maps `(X, seed, n_threads, budget) -> Y` (n × 2). `budget` scales the default
number of optimisation epochs (see `params.py`); `default_epochs` documents what ×1 means. Shared hyperparameters are
aligned across implementations: perplexity, n_neighbors and min_dist. Everything else stays
at library defaults. Library imports are lazy, so a worker process loads only the
library it benchmarks.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from drbench.algorithms import ncvis, pca, tsne, umap
from drbench.algorithms.params import TSNE_ITERATIONS, scaled_epochs

FitTransform = Callable[[np.ndarray, int, int, float], np.ndarray]


@dataclass(frozen=True)
class Algorithm:
    name: str
    family: str
    library: str
    language: str
    init: str
    fit_transform: FitTransform
    default_epochs: int | None

    @property
    def tunable(self) -> bool:
        return self.default_epochs is not None

    def epochs(self, budget: float) -> int | None:
        return scaled_epochs(self.default_epochs, budget) if self.tunable else None


# default_epochs = the main schedule length at budget ×1 (for N > 10k), for reporting.
REGISTRY: dict[str, Algorithm] = {
    algorithm.name: algorithm
    for algorithm in (
        Algorithm("pca", "pca", "scikit-learn", "python", "-", pca.sklearn_pca, None),
        Algorithm("opentsne-fft", "t-sne", "openTSNE", "python/c++", "pca", tsne.opentsne_fft,
                  TSNE_ITERATIONS),
        Algorithm("opentsne-fft-nndescent", "t-sne", "openTSNE", "python/c++", "pca",
                  tsne.opentsne_fft_nndescent,
                  TSNE_ITERATIONS),
        Algorithm("opentsne-fft-exact", "t-sne", "openTSNE", "python/c++", "pca",
                  tsne.opentsne_fft_exact,
                  TSNE_ITERATIONS),
        Algorithm("opentsne-fft-rinnd-fast", "t-sne", "openTSNE", "python/c++", "pca",
                  tsne.opentsne_fft_rinnd_fast,
                  TSNE_ITERATIONS),
        Algorithm("opentsne-fft-rinnd-default", "t-sne", "openTSNE", "python/c++", "pca",
                  tsne.opentsne_fft_rinnd_default, TSNE_ITERATIONS),
        Algorithm("mrs-tsne-bh", "t-sne", "manifolds-rs", "rust", "pca", tsne.manifolds_bh,
                  TSNE_ITERATIONS),
        Algorithm("mrs-tsne-fft", "t-sne", "manifolds-rs", "rust", "pca", tsne.manifolds_fft,
                  TSNE_ITERATIONS),
        Algorithm("mrs-tsne-fft-nndescent", "t-sne", "manifolds-rs", "rust", "pca",
                  tsne.manifolds_fft_nndescent, TSNE_ITERATIONS),
        Algorithm("mrs-tsne-fft-hnsw", "t-sne", "manifolds-rs", "rust", "pca",
                  tsne.manifolds_fft_hnsw, TSNE_ITERATIONS),
        Algorithm("mrs-tsne-fft-rinnd-fast", "t-sne", "manifolds-rs", "rust", "pca",
                  tsne.manifolds_fft_rinnd_fast, TSNE_ITERATIONS),
        Algorithm("mrs-tsne-fft-rinnd-default", "t-sne", "manifolds-rs", "rust", "pca",
                  tsne.manifolds_fft_rinnd_default, TSNE_ITERATIONS),
        Algorithm("mrs-tsne-bh-rinnd-fast", "t-sne", "manifolds-rs", "rust", "pca",
                  tsne.manifolds_bh_rinnd_fast, TSNE_ITERATIONS),
        Algorithm("mrs-tsne-bh-rinnd-default", "t-sne", "manifolds-rs", "rust", "pca",
                  tsne.manifolds_bh_rinnd_default, TSNE_ITERATIONS),
        Algorithm("mrs-tsne-qd-rinnd-fast", "t-sne", "manifolds-rs", "rust", "pca",
                  tsne.manifolds_qd_rinnd_fast, TSNE_ITERATIONS),
        Algorithm("mrs-tsne-qd-rinnd-default", "t-sne", "manifolds-rs", "rust", "pca",
                  tsne.manifolds_qd_rinnd_default, TSNE_ITERATIONS),
        Algorithm("umap-learn", "umap", "umap-learn", "python/numba", "recursive", umap.umap_learn,
                  umap.UMAP_LEARN_EPOCHS_LARGE),
        Algorithm("umap-learn-rinnd", "umap", "umap-learn", "python/rust", "recursive",
                  umap.umap_learn_rinnd, umap.UMAP_LEARN_EPOCHS_LARGE),
        Algorithm("umap-learn-rinnd-fast", "umap", "umap-learn", "python/rust", "recursive",
                  umap.umap_learn_rinnd_fast, umap.UMAP_LEARN_EPOCHS_LARGE),
        Algorithm("umap-learn-rinnd-default", "umap", "umap-learn", "python/rust", "recursive",
                  umap.umap_learn_rinnd_default, umap.UMAP_LEARN_EPOCHS_LARGE),
        Algorithm("umap-learn-legacy", "umap", "umap-learn", "python/numba", "spectral",
                  umap.umap_learn_legacy, umap.UMAP_LEARN_EPOCHS_LARGE),
        Algorithm("mrs-umap", "umap", "manifolds-rs", "rust", "spectral", umap.manifolds_umap,
                  umap.MANIFOLDS_UMAP_EPOCHS),
        Algorithm("mrs-umap-rinnd-fast", "umap", "manifolds-rs", "rust", "spectral",
                  umap.manifolds_umap_rinnd_fast, umap.MANIFOLDS_UMAP_EPOCHS),
        Algorithm("mrs-umap-rinnd-default", "umap", "manifolds-rs", "rust", "spectral",
                  umap.manifolds_umap_rinnd_default, umap.MANIFOLDS_UMAP_EPOCHS),
        Algorithm("mrs-pacmap", "pacmap", "manifolds-rs", "rust", "pca", umap.manifolds_pacmap,
                  umap.MANIFOLDS_PACMAP_EPOCHS),
        Algorithm("mrs-pacmap-rinnd-fast", "pacmap", "manifolds-rs", "rust", "pca",
                  umap.manifolds_pacmap_rinnd_fast, umap.MANIFOLDS_PACMAP_EPOCHS),
        Algorithm("mrs-pacmap-rinnd-default", "pacmap", "manifolds-rs", "rust", "pca",
                  umap.manifolds_pacmap_rinnd_default, umap.MANIFOLDS_PACMAP_EPOCHS),
        Algorithm("ncvis", "ncvis", "ncvis", "c++", "random", ncvis.ncvis, ncvis.NCVIS_EPOCHS),
        Algorithm("ncvis-ef64", "ncvis", "ncvis", "c++", "random", ncvis.ncvis_ef64,
                  ncvis.NCVIS_EPOCHS),
        Algorithm("ncvis-rinnd-fast", "ncvis", "ncvis", "c++", "random", ncvis.ncvis_rinnd_fast,
                  ncvis.NCVIS_EPOCHS),
        Algorithm("ncvis-rinnd-default", "ncvis", "ncvis", "c++", "random",
                  ncvis.ncvis_rinnd_default, ncvis.NCVIS_EPOCHS),
        Algorithm("ncvis-hq", "ncvis", "ncvis", "c++", "random", ncvis.ncvis_hq,
                  ncvis.NCVIS_EPOCHS),
    )
}


def get(name: str) -> Algorithm:
    if name not in REGISTRY:
        raise KeyError(f"unknown algorithm {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]
