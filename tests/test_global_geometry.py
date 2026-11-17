"""Global scores against brute-force definitions and sanity limits."""

import numpy as np
import pytest

from drbench.metrics import global_geometry as gg

N_POINTS = 500


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    centers = rng.standard_normal((6, 10)) * 5
    X = centers[rng.integers(6, size=N_POINTS)] + rng.standard_normal((N_POINTS, 10))
    return X


def test_scaled_copy_scores_one(data):
    X = data
    Y = 3.0 * X
    assert gg.pair_spearman(gg.sample_pairs(X, 1000), Y) == pytest.approx(1.0)
    assert gg.triplet_accuracy(gg.sample_triplets(X, 10_000), Y) == pytest.approx(1.0)


def test_pairs_never_self(data):
    pairs = gg.sample_pairs(data, 10_000)
    assert np.all(pairs.first != pairs.second)


def test_triplets_are_three_distinct_points(data):
    triplets = gg.sample_triplets(data, 20_000)
    assert np.all(triplets.anchor != triplets.first)
    assert np.all(triplets.anchor != triplets.second)
    assert np.all(triplets.first != triplets.second)


def test_triplet_accuracy_matches_brute_force(data):
    X = data
    triplets = gg.sample_triplets(X, 2_000)
    Y = X[:, :2] + np.random.default_rng(3).standard_normal((N_POINTS, 2))
    def closer(Z, i, j, l):
        return np.linalg.norm(Z[i] - Z[j]) < np.linalg.norm(Z[i] - Z[l])
    expected = np.mean([
        closer(X, i, j, l) == closer(Y, i, j, l)
        for i, j, l in zip(triplets.anchor, triplets.first, triplets.second)
    ])
    assert gg.triplet_accuracy(triplets, Y) == pytest.approx(expected, abs=1e-12)
