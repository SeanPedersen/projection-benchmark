"""NCVis+ adapter (https://github.com/SeanPedersen/ncvis-plus): NCVis with PCA init, a prefetching
layout kernel and retuned defaults. Defaults: 50 epochs, mean 10 negatives per edge (`-n20`: 20).

NCVis+ runs with its own defaults, including `min_dist` 0.001 instead of the shared 0.1. Its
tuning chose that value, so aligning it would benchmark a configuration nobody would run.
`n_neighbors` stays at the shared 15.
Without `graph`, NCVis+ builds its own RINND graph (RINND's default schedule, the same build as
`-rinnd-default`, but not shared or cached). With `graph`, it embeds the shared cached graph.
It takes indices only; the distances are not needed.
"""

from functools import partial

import numpy as np

from drbench.algorithms import knn_graph
from drbench.algorithms.params import N_NEIGHBORS, scaled_epochs

NCVIS_PLUS_EPOCHS = 50
DEFAULT_NEGATIVES = 10
QUALITY_NEGATIVES = 20


def _ncvis_plus(
    X: np.ndarray, seed: int, n_threads: int, budget: float, *,
    n_negative: float = DEFAULT_NEGATIVES, graph: str | None = None,
) -> np.ndarray:
    from ncvis_plus import NCVisPlus

    model = NCVisPlus(n_neighbors=N_NEIGHBORS, n_epochs=scaled_epochs(NCVIS_PLUS_EPOCHS, budget),
                      n_negative=n_negative, n_jobs=n_threads, random_state=seed)
    if graph is None:
        return model.fit_transform(X)
    indices, _ = knn_graph.rinnd_graph(X, N_NEIGHBORS, n_threads, graph)
    return model.fit_transform(X, knn_indices=indices)


ncvis_plus = _ncvis_plus
ncvis_plus_rinnd_default = partial(_ncvis_plus, graph=knn_graph.DEFAULT)
ncvis_plus_n20_rinnd_default = partial(_ncvis_plus, n_negative=QUALITY_NEGATIVES,
                                       graph=knn_graph.DEFAULT)
