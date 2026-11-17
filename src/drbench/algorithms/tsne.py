"""t-SNE adapters: openTSNE (FFT) and manifolds-rs (Barnes-Hut and FFT), plus FFT variants that
swap only the kNN backend. `mrs-tsne-fft-rinnd-fast` builds the graph with RINND and hands it to
manifolds-rs as a precomputed kNN graph (same layout as its own: no self-neighbour, plain
euclidean distances).

Default schedules:
- openTSNE: 250 early-exaggeration iterations, then 500 more.
- manifolds-rs: 1000 epochs in total, of which the first 250 use early exaggeration.
"""

from functools import partial

import numpy as np

from drbench.algorithms.params import PERPLEXITY, scaled_epochs

OPENTSNE_EARLY_EXAGGERATION_ITER = 250
OPENTSNE_ITER = 500
MANIFOLDS_TSNE_EPOCHS = 1000
MANIFOLDS_EARLY_EXAGGERATION_ITER = 250
# kNN backends the libraries pick on their own: openTSNE "auto" resolves to Annoy for euclidean
# data, manifolds-rs t-SNE defaults to IVF. The variants below swap only this backend.
OPENTSNE_DEFAULT_NEIGHBORS = "auto"
MANIFOLDS_TSNE_DEFAULT_ANN = "ivf"
TSNE_NEIGHBORS = round(3 * PERPLEXITY)
# Cheaper RINND schedule: on 100k x 512 at k=90 (8 threads), 9 s and recall 0.90 vs 31 s and
# 0.994 for the defaults, which matches manifolds-rs IVF's recall (0.88, ~54 s).
RINND_FAST = {"max_candidates": 20, "n_iters": 4}


def _opentsne(
    X: np.ndarray, seed: int, n_threads: int, budget: float, *, neighbors: str
) -> np.ndarray:
    from openTSNE import TSNE

    model = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        early_exaggeration_iter=scaled_epochs(OPENTSNE_EARLY_EXAGGERATION_ITER, budget),
        n_iter=scaled_epochs(OPENTSNE_ITER, budget),
        negative_gradient_method="fft",
        neighbors=neighbors,
        n_jobs=n_threads,
        random_state=seed,
    )
    return np.asarray(model.fit(X))


opentsne_fft = partial(_opentsne, neighbors=OPENTSNE_DEFAULT_NEIGHBORS)
opentsne_fft_nndescent = partial(_opentsne, neighbors="pynndescent")
opentsne_fft_exact = partial(_opentsne, neighbors="exact")


def _rinnd_graph(
    X: np.ndarray, seed: int, n_threads: int, rinnd_params: dict
) -> tuple[np.ndarray, np.ndarray]:
    from rinnd import RINND

    index = RINND(
        np.ascontiguousarray(X, dtype=np.float32),
        n_neighbors=TSNE_NEIGHBORS,
        n_jobs=n_threads,
        random_state=seed,
        graph_only=True,
        **rinnd_params,
    )
    indices, distances = index.neighbor_graph
    return indices.astype(np.int64), distances


def _manifolds_tsne(
    X: np.ndarray,
    seed: int,
    n_threads: int,
    budget: float,
    *,
    approx: str,
    ann: str,
    rinnd_params: dict | None = None,
) -> np.ndarray:
    import manifolds_rs as mf

    mf.set_num_threads(n_threads)
    knn = {}
    if rinnd_params is not None:
        knn_indices, knn_distances = _rinnd_graph(X, seed, n_threads, rinnd_params)
        knn = {"knn_indices": knn_indices, "knn_distances": knn_distances}
    model = mf.TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        n_epochs=scaled_epochs(MANIFOLDS_TSNE_EPOCHS, budget),
        approx=approx,
        ann=ann,
        seed=seed,
        optim_params=mf.TsneOptim(
            early_exag_iter=scaled_epochs(MANIFOLDS_EARLY_EXAGGERATION_ITER, budget)
        ),
    )
    return model.fit_transform(X, **knn)


manifolds_bh = partial(_manifolds_tsne, approx="barnes_hut", ann=MANIFOLDS_TSNE_DEFAULT_ANN)
manifolds_fft = partial(_manifolds_tsne, approx="fft", ann=MANIFOLDS_TSNE_DEFAULT_ANN)
manifolds_fft_nndescent = partial(_manifolds_tsne, approx="fft", ann="nndescent")
manifolds_fft_hnsw = partial(_manifolds_tsne, approx="fft", ann="hnsw")
manifolds_fft_rinnd_fast = partial(
    _manifolds_tsne, approx="fft", ann=MANIFOLDS_TSNE_DEFAULT_ANN, rinnd_params=RINND_FAST
)
