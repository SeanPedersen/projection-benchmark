"""Every registered algorithm returns a finite (N, 2) embedding on small data."""

import numpy as np
import pytest

from drbench import algorithms

N_POINTS, N_FEATURES, N_THREADS = 1000, 32, 2
BUDGET = 0.25


@pytest.fixture(scope="module")
def X():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((5, N_FEATURES)) * 4
    return (centers[rng.integers(5, size=N_POINTS)] + rng.standard_normal((N_POINTS, N_FEATURES))).astype(np.float32)


@pytest.mark.parametrize("name", sorted(algorithms.REGISTRY))
def test_embedding_shape_and_finite(X, name):
    Y = np.asarray(algorithms.get(name).fit_transform(X, 0, N_THREADS, BUDGET))
    assert Y.shape == (N_POINTS, 2)
    assert np.isfinite(Y).all()
