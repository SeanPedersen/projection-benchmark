"""Datasets derived from another registered dataset, so they need no extra download."""

from collections.abc import Callable
from pathlib import Path

import numpy as np

PCA_SEED = 0


def pca_projection(source_prepare: Callable[[Path], None], source_name: str, n_components: int):
    """Build a `prepare(target_dir)` that PCA-projects `source_name` (a sibling directory) down
    to `n_components` dimensions. Same points and cluster structure, far lower intrinsic cost
    for kNN search."""

    def prepare(target_dir: Path) -> None:
        from sklearn.decomposition import PCA

        source_dir = target_dir.parent / source_name
        source_path = source_dir / "X.npy"
        if not source_path.exists():
            source_dir.mkdir(parents=True, exist_ok=True)
            source_prepare(source_dir)
        X = np.load(source_path)
        Y = PCA(n_components, svd_solver="randomized", random_state=PCA_SEED).fit_transform(X)
        np.save(target_dir / "X.npy", np.ascontiguousarray(Y, dtype=np.float32))

    return prepare
