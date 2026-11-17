"""Global geometry scores: do distances between far-apart points survive the projection?

Both scores use fixed random samples, so their sampling noise is small and measured (sd on a fixed
embedding of 100k points: 0.001 for 1M pairs, 0.0002-0.0005 for 1M triplets), well below the
spread across algorithm seeds. k-means based scores (centroid/medoid ρ) were dropped: refitting
k-means alone moved them by 0.01-0.08.

- random pairs (headline `global`): Spearman ρ between original and 2D distances of uniformly
  sampled point pairs.
- random triplets (PaCMAP): share of sampled triplets (i, j, l) where "is j closer to i than l?"
  has the same answer in both spaces. Only relative order counts, so the long tail of large
  distances does not dominate.

k-means labels are still fitted once per dataset, but only to colour the embedding thumbnails.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

KMEANS_SEED = 0
KMEANS_RESTARTS = 3
PAIR_SAMPLE_SEED = 0
TRIPLET_SAMPLE_SEED = 0
PAIR_DISTANCE_CHUNK = 20_000


@dataclass(frozen=True)
class PairReference:
    first: np.ndarray
    second: np.ndarray
    distances: np.ndarray


@dataclass(frozen=True)
class TripletReference:
    anchor: np.ndarray
    first: np.ndarray
    second: np.ndarray
    first_closer: np.ndarray


def cluster_labels(X: np.ndarray, n_clusters: int) -> np.ndarray:
    kmeans = KMeans(n_clusters=n_clusters, n_init=KMEANS_RESTARTS, random_state=KMEANS_SEED)
    return kmeans.fit(X).labels_.astype(np.int64)


def rowwise_distances(X: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.concatenate([
        np.linalg.norm(X[first[s : s + PAIR_DISTANCE_CHUNK]] - X[second[s : s + PAIR_DISTANCE_CHUNK]], axis=1)
        for s in range(0, len(first), PAIR_DISTANCE_CHUNK)
    ])


def sample_pairs(X: np.ndarray, n_pairs: int, seed: int = PAIR_SAMPLE_SEED) -> PairReference:
    rng = np.random.default_rng(seed)
    first = rng.integers(len(X), size=n_pairs)
    offset = rng.integers(1, len(X), size=n_pairs)
    second = (first + offset) % len(X)
    return PairReference(first, second, rowwise_distances(X, first, second))


def sample_triplets(X: np.ndarray, n_triplets: int, seed: int = TRIPLET_SAMPLE_SEED) -> TripletReference:
    """Uniform triplets of three distinct points; needs at least three points."""
    n_points = len(X)
    rng = np.random.default_rng(seed)
    anchor = rng.integers(n_points, size=n_triplets)
    # Two distinct non-zero offsets give two distinct points, both different from the anchor.
    first_offset = rng.integers(1, n_points, size=n_triplets)
    second_offset = rng.integers(1, n_points - 1, size=n_triplets)
    second_offset += second_offset >= first_offset
    first = (anchor + first_offset) % n_points
    second = (anchor + second_offset) % n_points
    first_closer = rowwise_distances(X, anchor, first) < rowwise_distances(X, anchor, second)
    return TripletReference(anchor, first, second, first_closer)


def pair_spearman(reference: PairReference, Y: np.ndarray) -> float:
    ld_distances = rowwise_distances(Y, reference.first, reference.second)
    return float(spearmanr(reference.distances, ld_distances).statistic)


def triplet_accuracy(reference: TripletReference, Y: np.ndarray) -> float:
    first_closer = (rowwise_distances(Y, reference.anchor, reference.first)
                    < rowwise_distances(Y, reference.anchor, reference.second))
    return float(np.mean(first_closer == reference.first_closer))
