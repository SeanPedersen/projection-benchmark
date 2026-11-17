"""t-SNE adapters: openTSNE (FFT) and manifolds-rs (Barnes-Hut and FFT), plus FFT variants that
swap only the kNN backend.

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


def _manifolds_tsne(
    X: np.ndarray, seed: int, n_threads: int, budget: float, *, approx: str, ann: str
) -> np.ndarray:
    import manifolds_rs as mf

    mf.set_num_threads(n_threads)
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
    return model.fit_transform(X)


manifolds_bh = partial(_manifolds_tsne, approx="barnes_hut", ann=MANIFOLDS_TSNE_DEFAULT_ANN)
manifolds_fft = partial(_manifolds_tsne, approx="fft", ann=MANIFOLDS_TSNE_DEFAULT_ANN)
manifolds_fft_nndescent = partial(_manifolds_tsne, approx="fft", ann="nndescent")
manifolds_fft_hnsw = partial(_manifolds_tsne, approx="fft", ann="hnsw")
