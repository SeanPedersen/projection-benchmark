"""Evaluate an embedding against a per-dataset reference.

The reference holds the original-space artefacts that every embedding of a dataset shares:
the exact kNN, the sampled pairs and triplets, and k-means labels (thumbnail colours only). It is
computed once and cached in `data/<spec>/reference.npz`; parts that are missing or sized
differently from the config are recomputed on load, the rest is kept.

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


@dataclass(frozen=True)
class Reference:
    knn: np.ndarray
    color_labels: np.ndarray
    pairs: gg.PairReference
    triplets: gg.TripletReference


def build_reference(X: np.ndarray, config: MetricsConfig) -> Reference:
    return Reference(
        knn=exact_knn(X, config.k_max, config.chunk_rows),
        color_labels=gg.cluster_labels(X, config.color_clusters),
        pairs=gg.sample_pairs(X, config.n_random_pairs),
        triplets=gg.sample_triplets(X, config.n_triplets),
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
    extended = not (labels_cached and pairs_cached and triplets_cached)
    reference = Reference(knn=data["knn"], color_labels=color_labels, pairs=pairs, triplets=triplets)
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


ANCHOR_SEED = 0


def anchor_rows(n_points: int, n_anchors: int | None) -> np.ndarray | None:
    """Fixed random anchor sample for local scores, or None for all points."""
    if not n_anchors or n_anchors >= n_points:
        return None
    return np.sort(np.random.default_rng(ANCHOR_SEED).choice(n_points, n_anchors, replace=False))


def evaluate(X: np.ndarray, Y: np.ndarray, reference: Reference, config: MetricsConfig) -> dict[str, float]:
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    n_points = len(X)
    rows = anchor_rows(n_points, config.anchor_rows)
    hd_neighbors = reference.knn if rows is None else reference.knn[rows]
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
