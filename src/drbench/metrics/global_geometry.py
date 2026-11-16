"""Global geometry scores: do distances between far-apart regions survive the projection?

- centroid: k-means on the original vectors; Spearman ρ between pairwise distances of the
  k-means centers and of the per-cluster means in 2D.
- medoid: same, using each cluster's member nearest its center. This is robust when a method
  splits one cluster into several islands, which drags the 2D mean into empty space.
- random pairs: Spearman ρ between original and 2D distances of uniformly sampled point pairs
  (does not depend on clustering).
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

KMEANS_SEED = 0
KMEANS_RESTARTS = 3
PAIR_SAMPLE_SEED = 0
PAIR_DISTANCE_CHUNK = 20_000


@dataclass(frozen=True)
class ClusterReference:
    labels: np.ndarray
    center_distances: np.ndarray
    medoids: np.ndarray
    medoid_distances: np.ndarray

    @property
    def n_clusters(self) -> int:
        return len(self.medoids)


@dataclass(frozen=True)
class PairReference:
    first: np.ndarray
    second: np.ndarray
    distances: np.ndarray


def nearest_members(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distance_to_own_center = np.linalg.norm(X - centers[labels], axis=1)
    return np.array(
        [np.flatnonzero(labels == c)[np.argmin(distance_to_own_center[labels == c])]
         for c in range(len(centers))]
    )


def fit_clusters(X: np.ndarray, n_clusters: int) -> ClusterReference:
    kmeans = KMeans(n_clusters=n_clusters, n_init=KMEANS_RESTARTS, random_state=KMEANS_SEED).fit(X)
    medoids = nearest_members(X, kmeans.labels_, kmeans.cluster_centers_)
    return ClusterReference(
        labels=kmeans.labels_.astype(np.int64),
        center_distances=pdist(kmeans.cluster_centers_),
        medoids=medoids,
        medoid_distances=pdist(X[medoids]),
    )


def rowwise_distances(X: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.concatenate([
        np.linalg.norm(X[first[s : s + PAIR_DISTANCE_CHUNK]] - X[second[s : s + PAIR_DISTANCE_CHUNK]], axis=1)
        for s in range(0, len(first), PAIR_DISTANCE_CHUNK)
    ])


def sample_pairs(X: np.ndarray, n_pairs: int) -> PairReference:
    rng = np.random.default_rng(PAIR_SAMPLE_SEED)
    first = rng.integers(len(X), size=n_pairs)
    offset = rng.integers(1, len(X), size=n_pairs)
    second = (first + offset) % len(X)
    return PairReference(first, second, rowwise_distances(X, first, second))


def cluster_means(Y: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    sums = np.zeros((n_clusters, Y.shape[1]))
    np.add.at(sums, labels, Y)
    return sums / np.bincount(labels, minlength=n_clusters)[:, None]


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def centroid_spearman(reference: ClusterReference, Y: np.ndarray) -> float:
    means = cluster_means(Y, reference.labels, reference.n_clusters)
    return rank_correlation(reference.center_distances, pdist(means))


def medoid_spearman(reference: ClusterReference, Y: np.ndarray) -> float:
    return rank_correlation(reference.medoid_distances, pdist(Y[reference.medoids]))


def pair_spearman(reference: PairReference, Y: np.ndarray) -> float:
    return rank_correlation(reference.distances, rowwise_distances(Y, reference.first, reference.second))
