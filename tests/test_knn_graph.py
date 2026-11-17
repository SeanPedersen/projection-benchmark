"""Shared RINND graphs: layout, caching, and charging the stored build cost on reuse."""

import numpy as np
import pytest

from drbench.algorithms import knn_graph

K = 15


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(knn_graph, "_charged", knn_graph.ChargedCost())
    knn_graph.enable_cache(tmp_path, baseline_rss_kb=0)
    yield tmp_path
    monkeypatch.setattr(knn_graph, "_cache_dir", None)


def test_graph_layout():
    X = np.random.default_rng(0).standard_normal((600, 16)).astype(np.float32)
    indices, distances = knn_graph.rinnd_graph(X, K, n_threads=2)
    assert indices.shape == distances.shape == (600, K)
    assert not np.any(indices == np.arange(600)[:, None])
    assert np.all(np.diff(distances, axis=1) >= 0)
    with_self, self_distances = knn_graph.with_self(indices, distances, K)
    assert with_self.shape == (600, K)
    np.testing.assert_array_equal(with_self[:, 0], np.arange(600))
    assert np.all(self_distances[:, 0] == 0)


def test_reuse_charges_stored_build_cost(fresh_cache):
    X = np.random.default_rng(1).standard_normal((600, 16)).astype(np.float32)
    assert knn_graph.charged_cost().knn_s is None
    built = knn_graph.rinnd_graph(X, K, n_threads=2)
    assert knn_graph.charged_cost().build_s == 0.0
    stored = np.load(next(fresh_cache.glob("*.npz")))
    assert knn_graph.charged_cost().knn_s == pytest.approx(float(stored["build_s"]))
    loaded = knn_graph.rinnd_graph(X, K, n_threads=2)
    np.testing.assert_array_equal(loaded[0], built[0])
    assert knn_graph.charged_cost().build_s == pytest.approx(float(stored["build_s"]))
    assert knn_graph.charged_cost().extra_rss_kb == int(stored["extra_rss_kb"])
    assert knn_graph.charged_cost().knn_s == pytest.approx(2 * float(stored["build_s"]))


def test_schedules_that_coincide_share_one_cached_graph(tmp_path):
    fast, default = knn_graph.FAST, knn_graph.DEFAULT
    assert knn_graph.graph_path(tmp_path, 15, 100, 4, fast) == knn_graph.graph_path(tmp_path, 15, 100, 4, default)
    assert knn_graph.graph_path(tmp_path, 90, 100, 4, fast) != knn_graph.graph_path(tmp_path, 90, 100, 4, default)
    assert knn_graph.schedule_for(90, default) == {}
