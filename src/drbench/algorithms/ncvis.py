"""NCVis adapter (C++ core, Python wrapper). Defaults: 50 epochs plus 20 init epochs.

The kNN graph comes from HNSW. The stock settings (M=8, ef_construction=100, query ef=k+1) reach
only ~0.85 recall@15 on 512-d embeddings, so two variants raise it: `ef64` (stock graph, wider
query, ~0.94) and `hq` (denser graph and wider query, ~0.99). `ef_search` is a drbench patch to
the vendored source.
"""

from functools import partial

import numpy as np

from drbench.algorithms.params import MIN_DIST, N_NEIGHBORS, scaled_epochs

NCVIS_EPOCHS = 50
NCVIS_INIT_EPOCHS = 20
STOCK_M = 8
STOCK_EF_CONSTRUCTION = 100
WIDE_EF_SEARCH = 64
DENSE_M = 16
DENSE_EF_CONSTRUCTION = 200


def _ncvis(
    X: np.ndarray,
    seed: int,
    n_threads: int,
    budget: float,
    *,
    m: int = STOCK_M,
    ef_construction: int = STOCK_EF_CONSTRUCTION,
    ef_search: int | None = None,
) -> np.ndarray:
    from ncvis import NCVis

    model = NCVis(
        d=2,
        n_threads=n_threads,
        n_neighbors=N_NEIGHBORS,
        M=m,
        ef_construction=ef_construction,
        ef_search=ef_search,
        min_dist=MIN_DIST,
        n_epochs=scaled_epochs(NCVIS_EPOCHS, budget),
        n_init_epochs=scaled_epochs(NCVIS_INIT_EPOCHS, budget),
        random_seed=seed,
    )
    return model.fit_transform(X)


ncvis = _ncvis
ncvis_ef64 = partial(_ncvis, ef_search=WIDE_EF_SEARCH)
ncvis_hq = partial(
    _ncvis, m=DENSE_M, ef_construction=DENSE_EF_CONSTRUCTION, ef_search=WIDE_EF_SEARCH
)
