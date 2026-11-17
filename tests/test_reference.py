"""The cached reference round-trips and is extended in place when new parts are configured."""

from dataclasses import replace

import numpy as np

from drbench.config import MetricsConfig
from drbench.metrics import METRICS_VERSION, anchor_rows, evaluate, load_reference, reference_for

CONFIG = MetricsConfig(k_local=(5,), color_clusters=3, recall_k=5, n_random_pairs=500,
                       n_triplets=800, chunk_rows=64)


def test_reference_round_trip_and_extension(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 8))
    path = tmp_path / "reference.npz"
    built = reference_for(X, path, CONFIG)

    # A reference cached before triplets existed gets them added on load.
    np.savez(path, **{k: v for k, v in np.load(path).items() if not k.startswith("triplet_")})
    loaded, extended = load_reference(path, X, CONFIG)
    assert extended
    np.testing.assert_array_equal(loaded.triplets.first_closer, built.triplets.first_closer)

    reference_for(X, path, CONFIG)
    _, extended = load_reference(path, X, CONFIG)
    assert not extended

    scores = evaluate(X, X[:, :2], loaded, CONFIG)
    assert scores["metrics_version"] == METRICS_VERSION
    assert scores["global"] == scores["pair_rho"]
    assert 0.0 <= scores["triplet_acc"] <= 1.0


ANCHORED = replace(CONFIG, anchor_rows=50, anchors_above_n=100)


def test_anchor_rule_applies_only_above_threshold():
    assert anchor_rows(100, ANCHORED) is None
    assert len(anchor_rows(101, ANCHORED)) == 50
    assert anchor_rows(101, CONFIG) is None
    assert anchor_rows(40, replace(ANCHORED, anchors_above_n=0)) is None


def test_anchored_reference_holds_anchor_knn_only(tmp_path):
    X = np.random.default_rng(1).standard_normal((200, 8))
    full = reference_for(X, tmp_path / "full.npz", CONFIG)
    anchored = reference_for(X, tmp_path / "anchored.npz", ANCHORED)
    rows = anchor_rows(len(X), ANCHORED)
    np.testing.assert_array_equal(anchored.knn_rows, rows)
    np.testing.assert_array_equal(anchored.knn, full.knn[rows])

    # Same scores whether the anchor neighbours come from a full or an anchor-only kNN.
    Y = X[:, :2]
    assert evaluate(X, Y, anchored, ANCHORED) == evaluate(X, Y, full, ANCHORED)

    # A full kNN in the cache serves anchors; an anchor-only one is rebuilt for full-N scoring.
    _, extended = load_reference(tmp_path / "full.npz", X, ANCHORED)
    assert not extended
    reloaded, extended = load_reference(tmp_path / "anchored.npz", X, CONFIG)
    assert extended and reloaded.knn_rows is None and len(reloaded.knn) == len(X)
