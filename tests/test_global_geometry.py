"""Global scores against brute-force definitions and sanity limits."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from drbench.metrics import global_geometry as gg

N_POINTS = 500


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    centers = rng.standard_normal((6, 10)) * 5
    X = centers[rng.integers(6, size=N_POINTS)] + rng.standard_normal((N_POINTS, 10))
    return X


def test_centroid_matches_brute_force(data):
    X = data
    reference = gg.fit_clusters(X, 5)
    Y = X[:, :2] + np.random.default_rng(2).standard_normal((N_POINTS, 2))
    means_hd = np.stack([X[reference.labels == c].mean(0) for c in range(5)])
    means_ld = np.stack([Y[reference.labels == c].mean(0) for c in range(5)])
    expected = spearmanr(pdist(means_hd), pdist(means_ld)).statistic
    assert gg.centroid_spearman(reference, Y) == pytest.approx(expected)


def test_scaled_copy_scores_one(data):
    X = data
    Y = 3.0 * X
    assert gg.centroid_spearman(gg.fit_clusters(X, 5), Y) == pytest.approx(1.0)
    assert gg.medoid_spearman(gg.fit_clusters(X, 5), Y) == pytest.approx(1.0)
    assert gg.pair_spearman(gg.sample_pairs(X, 1000), Y) == pytest.approx(1.0)


def test_pairs_never_self(data):
    pairs = gg.sample_pairs(data, 10_000)
    assert np.all(pairs.first != pairs.second)


def test_medoids_belong_to_their_cluster(data):
    reference = gg.fit_clusters(data, 5)
    assert np.array_equal(reference.labels[reference.medoids], np.arange(5))
