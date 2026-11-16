"""PCA baseline (scikit-learn). It has no iterations, so the budget is ignored."""

import numpy as np


def sklearn_pca(X: np.ndarray, seed: int, n_threads: int, budget: float) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=seed).fit_transform(X)
