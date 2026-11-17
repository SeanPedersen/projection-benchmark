"""t-SNE adapters: openTSNE (FFT) and manifolds-rs (Barnes-Hut and FFT), plus FFT variants that
swap only the kNN backend. The `-rinnd-fast` variants take the shared RINND graph
(`knn_graph.py`, k = 3 x perplexity) as a precomputed kNN graph.

Both libraries run the same schedule (`params.tsne_schedule`): 750 iterations, of which the first
250 use early exaggeration 12 (openTSNE's defaults; manifolds-rs would run 1000). Still different:
openTSNE's automatic learning rate is N / exaggeration per phase (N/12, then N), manifolds-rs uses
max(N/12, 200) throughout.
"""

from functools import partial

import numpy as np

from drbench.algorithms import knn_graph
from drbench.algorithms.params import (
    PERPLEXITY,
    TSNE_EARLY_EXAGGERATION,
    tsne_schedule,
)

# kNN backends the libraries pick on their own: openTSNE "auto" resolves to Annoy for euclidean
# data, manifolds-rs t-SNE defaults to IVF. The variants below swap only this backend.
OPENTSNE_DEFAULT_NEIGHBORS = "auto"
MANIFOLDS_TSNE_DEFAULT_ANN = "ivf"
TSNE_NEIGHBORS = round(3 * PERPLEXITY)


def _opentsne(
    X: np.ndarray, seed: int, n_threads: int, budget: float, *, neighbors: str
) -> np.ndarray:
    from openTSNE import TSNE
    from openTSNE.affinity import PerplexityBasedNN
    from openTSNE.nearest_neighbors import PrecomputedNeighbors

    affinities = None
    if neighbors == SHARED_RINND:
        indices, distances = knn_graph.rinnd_graph(X, TSNE_NEIGHBORS, n_threads)
        affinities = PerplexityBasedNN(knn_index=PrecomputedNeighbors(indices, distances),
                                       perplexity=PERPLEXITY, n_jobs=n_threads)
        neighbors = OPENTSNE_DEFAULT_NEIGHBORS
    early_iterations, total_iterations = tsne_schedule(budget)
    model = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        early_exaggeration_iter=early_iterations,
        early_exaggeration=TSNE_EARLY_EXAGGERATION,
        n_iter=total_iterations - early_iterations,
        negative_gradient_method="fft",
        neighbors=neighbors,
        n_jobs=n_threads,
        random_state=seed,
    )
    return np.asarray(model.fit(X, affinities=affinities))


SHARED_RINND = "shared-rinnd"
opentsne_fft = partial(_opentsne, neighbors=OPENTSNE_DEFAULT_NEIGHBORS)
opentsne_fft_nndescent = partial(_opentsne, neighbors="pynndescent")
opentsne_fft_exact = partial(_opentsne, neighbors="exact")
opentsne_fft_rinnd_fast = partial(_opentsne, neighbors=SHARED_RINND)


def _manifolds_tsne(
    X: np.ndarray,
    seed: int,
    n_threads: int,
    budget: float,
    *,
    approx: str,
    ann: str,
    shared_graph: bool = False,
) -> np.ndarray:
    import manifolds_rs as mf

    mf.set_num_threads(n_threads)
    early_iterations, total_iterations = tsne_schedule(budget)
    knn = {}
    if shared_graph:
        knn_indices, knn_distances = knn_graph.rinnd_graph(X, TSNE_NEIGHBORS, n_threads)
        knn = {"knn_indices": knn_indices, "knn_distances": knn_distances}
    model = mf.TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        n_epochs=total_iterations,
        approx=approx,
        ann=ann,
        seed=seed,
        optim_params=mf.TsneOptim(
            early_exag_iter=early_iterations, early_exag_factor=TSNE_EARLY_EXAGGERATION
        ),
    )
    return model.fit_transform(X, **knn)


manifolds_bh = partial(_manifolds_tsne, approx="barnes_hut", ann=MANIFOLDS_TSNE_DEFAULT_ANN)
manifolds_fft = partial(_manifolds_tsne, approx="fft", ann=MANIFOLDS_TSNE_DEFAULT_ANN)
manifolds_fft_nndescent = partial(_manifolds_tsne, approx="fft", ann="nndescent")
manifolds_fft_hnsw = partial(_manifolds_tsne, approx="fft", ann="hnsw")
manifolds_fft_rinnd_fast = partial(
    _manifolds_tsne, approx="fft", ann=MANIFOLDS_TSNE_DEFAULT_ANN, shared_graph=True
)
