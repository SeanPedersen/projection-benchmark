"""t-SNE adapters: openTSNE (FFT) and manifolds-rs (Barnes-Hut and FFT).

Default schedules:
- openTSNE: 250 early-exaggeration iterations, then 500 more.
- manifolds-rs: 1000 epochs in total, of which the first 250 use early exaggeration.
"""

import numpy as np

from drbench.algorithms.params import PERPLEXITY, scaled_epochs

OPENTSNE_EARLY_EXAGGERATION_ITER = 250
OPENTSNE_ITER = 500
MANIFOLDS_TSNE_EPOCHS = 1000
MANIFOLDS_EARLY_EXAGGERATION_ITER = 250


def opentsne_fft(X: np.ndarray, seed: int, n_threads: int, budget: float) -> np.ndarray:
    from openTSNE import TSNE

    model = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        early_exaggeration_iter=scaled_epochs(OPENTSNE_EARLY_EXAGGERATION_ITER, budget),
        n_iter=scaled_epochs(OPENTSNE_ITER, budget),
        negative_gradient_method="fft",
        n_jobs=n_threads,
        random_state=seed,
    )
    return np.asarray(model.fit(X))


def _manifolds_tsne(X: np.ndarray, seed: int, n_threads: int, budget: float, approx: str) -> np.ndarray:
    import manifolds_rs as mf

    mf.set_num_threads(n_threads)
    model = mf.TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        n_epochs=scaled_epochs(MANIFOLDS_TSNE_EPOCHS, budget),
        approx=approx,
        seed=seed,
        optim_params=mf.TsneOptim(
            early_exag_iter=scaled_epochs(MANIFOLDS_EARLY_EXAGGERATION_ITER, budget)
        ),
    )
    return model.fit_transform(X)


def manifolds_bh(X: np.ndarray, seed: int, n_threads: int, budget: float) -> np.ndarray:
    return _manifolds_tsne(X, seed, n_threads, budget, approx="barnes_hut")


def manifolds_fft(X: np.ndarray, seed: int, n_threads: int, budget: float) -> np.ndarray:
    return _manifolds_tsne(X, seed, n_threads, budget, approx="fft")
