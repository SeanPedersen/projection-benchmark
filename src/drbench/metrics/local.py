"""Local neighbourhood scores computed from exact neighbour ranks (see `neighbors.py`).

- trustworthiness(k): penalises 2D neighbours that were far apart originally (false neighbours).
- continuity(k): penalises original neighbours pushed apart in 2D (torn neighbourhoods).
- recall(k): share of the k nearest 2D neighbours that are among the k nearest originally.
"""

import numpy as np


def rank_penalty_score(ranks: np.ndarray, k: int, n_points: int | None = None) -> float:
    """1 - normalised sum of rank excess over k (Venna & Kaski); `ranks` are in the *other* space.

    `ranks` may cover only a sample of anchor points; then this is an unbiased estimate of the
    full-data score, with the normalisation taken from the full dataset size `n_points`.
    """
    n_anchors = len(ranks)
    n = n_anchors if n_points is None else n_points
    excess = np.maximum(ranks[:, :k] - k, 0).sum()
    return float(1.0 - 2.0 / (n_anchors * k * (2.0 * n - 3.0 * k - 1.0)) * excess)


def trustworthiness(ranks_hd_of_ld_neighbors: np.ndarray, k: int, n_points: int | None = None) -> float:
    return rank_penalty_score(ranks_hd_of_ld_neighbors, k, n_points)


def continuity(ranks_ld_of_hd_neighbors: np.ndarray, k: int, n_points: int | None = None) -> float:
    return rank_penalty_score(ranks_ld_of_hd_neighbors, k, n_points)


def knn_recall(ranks_hd_of_ld_neighbors: np.ndarray, k: int) -> float:
    return float((ranks_hd_of_ld_neighbors[:, :k] <= k).mean())
