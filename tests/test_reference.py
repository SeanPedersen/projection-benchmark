"""The cached reference round-trips and is extended in place when new parts are configured."""

import numpy as np

from drbench.config import MetricsConfig
from drbench.metrics import METRICS_VERSION, evaluate, load_reference, reference_for

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
