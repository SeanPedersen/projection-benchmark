"""Evaluate an embedding against a per-dataset reference.

The reference holds the original-space artefacts that every embedding of a dataset shares:
the exact kNN, the k-means clusterings and the sampled pairs. It is computed once and cached
in `data/<spec>/reference.npz`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drbench.config import MetricsConfig
from drbench.metrics import global_geometry as gg
from drbench.metrics import local
from drbench.metrics.neighbors import exact_knn, lowdim_knn, neighbor_ranks


@dataclass(frozen=True)
class Reference:
    knn: np.ndarray
    clusters: dict[int, gg.ClusterReference]
    pairs: gg.PairReference


def build_reference(X: np.ndarray, config: MetricsConfig) -> Reference:
    return Reference(
        knn=exact_knn(X, config.k_max, config.chunk_rows),
        clusters={k: gg.fit_clusters(X, k) for k in config.k_clusters},
        pairs=gg.sample_pairs(X, config.n_random_pairs),
    )


def save_reference(reference: Reference, path: Path) -> None:
    arrays = {"knn": reference.knn, "pair_first": reference.pairs.first,
              "pair_second": reference.pairs.second, "pair_distances": reference.pairs.distances}
    for k, c in reference.clusters.items():
        arrays |= {f"k{k}_labels": c.labels, f"k{k}_center_distances": c.center_distances,
                   f"k{k}_medoids": c.medoids, f"k{k}_medoid_distances": c.medoid_distances}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_reference(path: Path, config: MetricsConfig) -> Reference:
    data = np.load(path)
    clusters = {
        k: gg.ClusterReference(data[f"k{k}_labels"], data[f"k{k}_center_distances"],
                               data[f"k{k}_medoids"], data[f"k{k}_medoid_distances"])
        for k in config.k_clusters
    }
    pairs = gg.PairReference(data["pair_first"], data["pair_second"], data["pair_distances"])
    return Reference(knn=data["knn"], clusters=clusters, pairs=pairs)


def reference_for(X: np.ndarray, path: Path, config: MetricsConfig) -> Reference:
    if path.exists():
        return load_reference(path, config)
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

    scores: dict[str, float] = {"anchors": n_points if rows is None else len(rows)}
    for k in config.k_local:
        scores[f"trust_{k}"] = local.trustworthiness(ranks_hd, k, n_points)
        scores[f"continuity_{k}"] = local.continuity(ranks_ld, k, n_points)
    scores[f"recall_{config.recall_k}"] = local.knn_recall(ranks_hd, config.recall_k)
    for k, clusters in reference.clusters.items():
        scores[f"centroid_rho_{k}"] = gg.centroid_spearman(clusters, Y)
        scores[f"medoid_rho_{k}"] = gg.medoid_spearman(clusters, Y)
    scores["pair_rho"] = gg.pair_spearman(reference.pairs, Y)

    scores["local"] = float(np.mean([scores[f"trust_{k}"] for k in config.k_local]))
    scores["global"] = float(np.mean([scores[f"centroid_rho_{k}"] for k in config.k_clusters]))
    return scores
