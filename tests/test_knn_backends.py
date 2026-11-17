"""Every kNN backend returns (n, k) neighbour indices that exclude the point itself and
overlap strongly with the exact graph on small clustered data."""

import numpy as np
import pytest

from drbench.knn import backends
from drbench.knn.bench import recall_at_k

N_POINTS, N_FEATURES, N_NEIGHBORS, N_THREADS = 1500, 16, 10, 2
MIN_RECALL = 0.7


@pytest.fixture(scope="module")
def X():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((6, N_FEATURES)) * 4
    return (centers[rng.integers(6, size=N_POINTS)] + rng.standard_normal((N_POINTS, N_FEATURES))).astype(np.float32)


@pytest.fixture(scope="module")
def truth(X):
    return backends.get("mrs-exhaustive").build(X, N_NEIGHBORS, N_THREADS)


@pytest.mark.parametrize("name", sorted(backends.REGISTRY))
def test_backend_shape_and_recall(X, truth, name):
    found = np.asarray(backends.get(name).build(X, N_NEIGHBORS, N_THREADS))
    assert found.shape == (N_POINTS, N_NEIGHBORS)
    assert not (found == np.arange(N_POINTS)[:, None]).any()
    assert recall_at_k(found, truth) >= MIN_RECALL


def test_recall_of_identical_graphs_is_one(truth):
    assert recall_at_k(truth, truth) == 1.0
