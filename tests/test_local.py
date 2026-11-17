"""Local scores must match scikit-learn's reference trustworthiness exactly."""

import numpy as np
import pytest
from sklearn.manifold import trustworthiness as sklearn_trustworthiness

from drbench.metrics import local
from drbench.metrics.neighbors import exact_knn, lowdim_knn, neighbor_ranks

N_POINTS, N_FEATURES, K_MAX = 500, 20, 30
CHUNK_ROWS = 64


@pytest.fixture(scope="module")
def spaces():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_POINTS, N_FEATURES))
    Y = X[:, :2] + 0.5 * rng.standard_normal((N_POINTS, 2))
    return X, Y


def brute_force_ranks(space: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    distances = ((space[:, None, :] - space[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=1, kind="stable")
    rank_of = np.empty_like(order)
    np.put_along_axis(rank_of, order, np.arange(1, len(space) + 1)[None, :], axis=1)
    return np.take_along_axis(rank_of, neighbors, axis=1)


def test_ranks_match_brute_force(spaces):
    X, Y = spaces
    neighbors = lowdim_knn(Y, K_MAX)
    np.testing.assert_array_equal(
        neighbor_ranks(X, neighbors, CHUNK_ROWS), brute_force_ranks(X, neighbors)
    )


def test_exact_knn_matches_kd_tree(spaces):
    X, _ = spaces
    np.testing.assert_array_equal(exact_knn(X, K_MAX, CHUNK_ROWS), lowdim_knn(X, K_MAX))


@pytest.mark.parametrize("k", [5, 15, 30])
def test_trustworthiness_matches_sklearn(spaces, k):
    X, Y = spaces
    ranks = neighbor_ranks(X, lowdim_knn(Y, K_MAX), CHUNK_ROWS)
    expected = sklearn_trustworthiness(X, Y, n_neighbors=k)
    assert local.trustworthiness(ranks, k) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("k", [5, 15])
def test_continuity_is_trustworthiness_with_spaces_swapped(spaces, k):
    X, Y = spaces
    ranks = neighbor_ranks(Y, exact_knn(X, K_MAX, CHUNK_ROWS), CHUNK_ROWS)
    expected = sklearn_trustworthiness(Y, X, n_neighbors=k)
    assert local.continuity(ranks, k) == pytest.approx(expected, abs=1e-12)


def test_identity_embedding_is_perfect(spaces):
    X, _ = spaces
    ranks = neighbor_ranks(X, exact_knn(X, K_MAX, CHUNK_ROWS), CHUNK_ROWS)
    assert local.trustworthiness(ranks, 15) == pytest.approx(1.0)
    assert local.knn_recall(ranks, 15) == pytest.approx(1.0)


def test_anchor_sample_matches_full_ranks(spaces):
    X, Y = spaces
    rows = np.sort(np.random.default_rng(3).choice(N_POINTS, 100, replace=False))
    full_hd = neighbor_ranks(X, lowdim_knn(Y, K_MAX), CHUNK_ROWS)
    np.testing.assert_array_equal(
        neighbor_ranks(X, lowdim_knn(Y, K_MAX, rows), CHUNK_ROWS, rows), full_hd[rows]
    )
    hd_knn = exact_knn(X, K_MAX, CHUNK_ROWS)
    full_ld = neighbor_ranks(Y, hd_knn, CHUNK_ROWS)
    np.testing.assert_array_equal(neighbor_ranks(Y, hd_knn[rows], CHUNK_ROWS, rows), full_ld[rows])


def test_anchor_estimate_is_close_to_full_score(spaces):
    X, Y = spaces
    rows = np.sort(np.random.default_rng(4).choice(N_POINTS, 250, replace=False))
    full = local.trustworthiness(neighbor_ranks(X, lowdim_knn(Y, K_MAX), CHUNK_ROWS), 15)
    sampled = local.trustworthiness(
        neighbor_ranks(X, lowdim_knn(Y, K_MAX, rows), CHUNK_ROWS, rows), 15, N_POINTS
    )
    assert sampled == pytest.approx(full, abs=0.01)


def test_exact_knn_of_anchor_rows_matches_full(spaces):
    X, _ = spaces
    rows = np.array([3, 17, 250, 499])
    np.testing.assert_array_equal(exact_knn(X, K_MAX, CHUNK_ROWS, rows), exact_knn(X, K_MAX, CHUNK_ROWS)[rows])
