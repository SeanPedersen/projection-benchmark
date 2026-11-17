"""Evaluate an embedding against a per-dataset reference.

The reference holds the original-space artefacts that every embedding of a dataset shares:
the exact kNN, the sampled pairs and triplets, and k-means labels (thumbnail colours only). It is
computed once and cached in `data/<spec>/reference.npz`; parts that are missing or sized
differently from the config are recomputed on load, the rest is kept.

Above `anchors_above_n` points, local scores are estimated on `anchor_rows` fixed anchor points
(ranks stay exact over all N), and the reference kNN is computed for those anchors only: an exact
kNN of every point is an N² brute force. A full-N kNN in the cache serves any anchor set.

`local` = mean trustworthiness over `k_local`; `global` = random-pair Spearman ρ.
`METRICS_VERSION` changes whenever a score's definition does, so stored scores get recomputed.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drbench.config import MetricsConfig
from drbench.metrics import global_geometry as gg
from drbench.metrics import local
from drbench.metrics.neighbors import exact_knn, lowdim_knn, neighbor_ranks

METRICS_VERSION = 2


ANCHOR_SEED = 0


def anchor_rows(n_points: int, config: MetricsConfig) -> np.ndarray | None:
    """Fixed random anchor sample for local scores, or None for all points."""
    n_anchors = config.anchor_rows
    if not n_anchors or n_anchors >= n_points or n_points <= config.anchors_above_n:
        return None
    return np.sort(np.random.default_rng(ANCHOR_SEED).choice(n_points, n_anchors, replace=False))


@dataclass(frozen=True)
class Reference:
    """`knn[i]` are the exact neighbours of point `knn_rows[i]` (of point i if `knn_rows` is None)."""

    knn: np.ndarray
    color_labels: np.ndarray
    pairs: gg.PairReference
    triplets: gg.TripletReference
    knn_rows: np.ndarray | None = None

    def covers(self, rows: np.ndarray | None, k: int) -> bool:
        if self.knn.shape[1] < k:
            return False
        if self.knn_rows is None:
            return True
        return rows is not None and np.array_equal(rows, self.knn_rows)

    def neighbors_of(self, rows: np.ndarray | None, k: int) -> np.ndarray:
        if not self.covers(rows, k):
            raise ValueError("reference kNN does not cover the requested rows")
        knn = self.knn[:, :k]
        return knn if rows is None or self.knn_rows is not None else knn[rows]


def build_knn(X: np.ndarray, config: MetricsConfig) -> tuple[np.ndarray, np.ndarray | None]:
    rows = anchor_rows(len(X), config)
    return exact_knn(X, config.k_max, config.chunk_rows, rows), rows


def build_reference(X: np.ndarray, config: MetricsConfig) -> Reference:
    knn, knn_rows = build_knn(X, config)
    return Reference(
        knn=knn,
        color_labels=gg.cluster_labels(X, config.color_clusters),
        pairs=gg.sample_pairs(X, config.n_random_pairs),
        triplets=gg.sample_triplets(X, config.n_triplets),
        knn_rows=knn_rows,
    )


def labels_key(n_clusters: int) -> str:
    return f"k{n_clusters}_labels"


def save_reference(reference: Reference, path: Path) -> None:
    arrays = {"knn": reference.knn, "pair_first": reference.pairs.first,
              "pair_second": reference.pairs.second, "pair_distances": reference.pairs.distances,
              "triplet_anchor": reference.triplets.anchor, "triplet_first": reference.triplets.first,
              "triplet_second": reference.triplets.second,
              "triplet_first_closer": reference.triplets.first_closer}
    arrays[labels_key(len(np.unique(reference.color_labels)))] = reference.color_labels
    if reference.knn_rows is not None:
        arrays["knn_rows"] = reference.knn_rows
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_reference(path: Path, X: np.ndarray, config: MetricsConfig) -> tuple[Reference, bool]:
    """Cached reference plus whether parts missing from the file had to be computed."""
    data = dict(np.load(path))
    labels_cached = labels_key(config.color_clusters) in data
    color_labels = (data[labels_key(config.color_clusters)] if labels_cached
                    else gg.cluster_labels(X, config.color_clusters))
    pairs_cached = len(data["pair_first"]) == config.n_random_pairs
    pairs = gg.PairReference(
        data["pair_first"], data["pair_second"], data["pair_distances"]
    ) if pairs_cached else gg.sample_pairs(X, config.n_random_pairs)
    triplets_cached = len(data.get("triplet_anchor", ())) == config.n_triplets
    triplets = gg.TripletReference(
        data["triplet_anchor"], data["triplet_first"], data["triplet_second"],
        data["triplet_first_closer"],
    ) if triplets_cached else gg.sample_triplets(X, config.n_triplets)
    cached_knn = Reference(knn=data["knn"], color_labels=color_labels, pairs=pairs,
                           triplets=triplets, knn_rows=data.get("knn_rows"))
    knn_cached = cached_knn.covers(anchor_rows(len(X), config), config.k_max)
    knn, knn_rows = (cached_knn.knn, cached_knn.knn_rows) if knn_cached else build_knn(X, config)
    extended = not (labels_cached and pairs_cached and triplets_cached and knn_cached)
    reference = Reference(knn=knn, color_labels=color_labels, pairs=pairs, triplets=triplets,
                          knn_rows=knn_rows)
    return reference, extended


def reference_for(X: np.ndarray, path: Path, config: MetricsConfig) -> Reference:
    if path.exists():
        reference, extended = load_reference(path, X, config)
        if extended:
            save_reference(reference, path)
        return reference
    reference = build_reference(X, config)
    save_reference(reference, path)
    return reference


def evaluate(X: np.ndarray, Y: np.ndarray, reference: Reference, config: MetricsConfig) -> dict[str, float]:
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    n_points = len(X)
    rows = anchor_rows(n_points, config)
    hd_neighbors = reference.neighbors_of(rows, config.k_max)
    ranks_hd = neighbor_ranks(X, lowdim_knn(Y, config.k_max, rows), config.chunk_rows, rows)
    ranks_ld = neighbor_ranks(Y, hd_neighbors, config.chunk_rows, rows)

    scores: dict[str, float] = {"metrics_version": METRICS_VERSION,
                                "anchors": n_points if rows is None else len(rows)}
    for k in config.k_local:
        scores[f"trust_{k}"] = local.trustworthiness(ranks_hd, k, n_points)
        scores[f"continuity_{k}"] = local.continuity(ranks_ld, k, n_points)
    scores[f"recall_{config.recall_k}"] = local.knn_recall(ranks_hd, config.recall_k)
    scores["pair_rho"] = gg.pair_spearman(reference.pairs, Y)
    scores["triplet_acc"] = gg.triplet_accuracy(reference.triplets, Y)

    scores["local"] = float(np.mean([scores[f"trust_{k}"] for k in config.k_local]))
    scores["global"] = scores["pair_rho"]
    return scores
