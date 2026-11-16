"""Exact neighbour ranks at large N without an N×N distance matrix.

Rows are processed in chunks. For high-dimensional spaces, each chunk's squared distances to all
points come from one BLAS matmul. A numba kernel then counts, per row, how many points lie
strictly closer than each queried neighbour: one linear pass gives all k ranks, with a binary
search over the k sorted thresholds only for the few points inside the farthest one.
Low-dimensional spaces (the embeddings) skip the matrix and compute distances inside the kernel.
Memory is O(chunk_rows × N); time is O(N² · D).

Rank convention (as in sklearn): the nearest non-self point has rank 1.
"""

import numpy as np
from numba import njit, prange
from sklearn.neighbors import NearestNeighbors

FUSED_MAX_DIM = 8


def squared_distances(block: np.ndarray, points: np.ndarray, point_sq_norms: np.ndarray) -> np.ndarray:
    distances = block @ points.T
    distances *= -2.0
    distances += point_sq_norms[None, :]
    distances += np.einsum("ij,ij->i", block, block)[:, None]
    return distances


@njit(cache=True, inline="always")
def _bucket(thresholds, d):
    """Number of thresholds <= d (thresholds ascending)."""
    lo, hi = 0, len(thresholds)
    while lo < hi:
        mid = (lo + hi) // 2
        if thresholds[mid] <= d:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True, inline="always")
def _write_counts(histogram, counts_row):
    running = 0
    for j in range(len(counts_row)):
        running += histogram[j]
        counts_row[j] = running


@njit(parallel=True, cache=True)
def _count_strictly_closer(distances, sorted_thresholds, self_cols, counts):
    n_rows, n_points = distances.shape
    k = sorted_thresholds.shape[1]
    for row in prange(n_rows):
        thresholds = sorted_thresholds[row]
        farthest = thresholds[k - 1]
        self_col = self_cols[row]
        # histogram[p] = number of points d with exactly p thresholds <= d
        histogram = np.zeros(k + 1, np.int64)
        for col in range(n_points):
            d = distances[row, col]
            # Fast path: almost all points lie beyond the farthest threshold and never count.
            if d >= farthest or col == self_col:
                continue
            histogram[_bucket(thresholds, d)] += 1
        _write_counts(histogram, counts[row])


@njit(parallel=True, cache=True)
def _count_strictly_closer_fused(space, rows, neighbors, ranks):
    n_points, n_dims = space.shape
    k = neighbors.shape[1]
    for row in prange(len(rows)):
        point = rows[row]
        thresholds = np.empty(k)
        for j in range(k):
            other = neighbors[row, j]
            d = 0.0
            for dim in range(n_dims):
                delta = space[point, dim] - space[other, dim]
                d += delta * delta
            thresholds[j] = d
        order = np.argsort(thresholds)
        sorted_thresholds = thresholds[order]
        farthest = sorted_thresholds[k - 1]
        histogram = np.zeros(k + 1, np.int64)
        for col in range(n_points):
            d = 0.0
            for dim in range(n_dims):
                delta = space[point, dim] - space[col, dim]
                d += delta * delta
            if d >= farthest or col == point:
                continue
            histogram[_bucket(sorted_thresholds, d)] += 1
        counts = np.empty(k, np.int64)
        _write_counts(histogram, counts)
        for j in range(k):
            ranks[row, order[j]] = counts[j] + 1


def all_rows(space: np.ndarray) -> np.ndarray:
    return np.arange(len(space))


def neighbor_ranks(
    space: np.ndarray, neighbors: np.ndarray, chunk_rows: int, rows: np.ndarray | None = None
) -> np.ndarray:
    """Rank of `neighbors[r, j]` among all points ordered by distance from point `rows[r]`.

    `rows` (default: all points) restricts the work to anchor points; ranks stay exact
    with respect to the full dataset.
    """
    rows = all_rows(space) if rows is None else rows
    ranks = np.empty(neighbors.shape, dtype=np.int64)
    if space.shape[1] <= FUSED_MAX_DIM:
        _count_strictly_closer_fused(
            np.ascontiguousarray(space, dtype=np.float64), rows, neighbors, ranks
        )
        return ranks
    sq_norms = np.einsum("ij,ij->i", space, space)
    for start in range(0, len(rows), chunk_rows):
        stop = min(start + chunk_rows, len(rows))
        distances = squared_distances(space[rows[start:stop]], space, sq_norms)
        thresholds = np.take_along_axis(distances, neighbors[start:stop], axis=1)
        order = np.argsort(thresholds, axis=1)
        counts = np.empty(order.shape, dtype=np.int64)
        _count_strictly_closer(
            distances, np.take_along_axis(thresholds, order, axis=1), rows[start:stop], counts
        )
        np.put_along_axis(ranks[start:stop], order, counts + 1, axis=1)
    return ranks


def drop_self(indices: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Remove each row's own index from k+1 neighbour lists (or the farthest one if absent,
    which happens with exact duplicates)."""
    n_rows, width = indices.shape
    is_self = indices == rows[:, None]
    is_self[~is_self.any(axis=1), -1] = True
    return indices[~is_self].reshape(n_rows, width - 1)


def exact_knn(space: np.ndarray, k: int, chunk_rows: int) -> np.ndarray:
    """Exact k nearest neighbours (self excluded), sorted by distance. Brute force in chunks."""
    n_points = len(space)
    sq_norms = np.einsum("ij,ij->i", space, space)
    knn = np.empty((n_points, k + 1), dtype=np.int64)
    for start in range(0, n_points, chunk_rows):
        stop = min(start + chunk_rows, n_points)
        distances = squared_distances(space[start:stop], space, sq_norms)
        candidates = np.argpartition(distances, k, axis=1)[:, : k + 1]
        order = np.argsort(np.take_along_axis(distances, candidates, axis=1), axis=1)
        knn[start:stop] = np.take_along_axis(candidates, order, axis=1)
    return drop_self(knn, all_rows(space))


def lowdim_knn(Y: np.ndarray, k: int, rows: np.ndarray | None = None) -> np.ndarray:
    """Exact kNN of `rows` (default: all) in a low-dimensional embedding via a KD-tree
    (self excluded, sorted)."""
    rows = all_rows(Y) if rows is None else rows
    index = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(Y)
    return drop_self(index.kneighbors(Y[rows], return_distance=False), rows)
