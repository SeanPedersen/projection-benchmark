"""Hyperparameters shared across implementations and the iteration-budget scaling.

A budget multiplier scales an algorithm's own default number of optimisation epochs (×1 = the
library default). Multi-phase schedules (t-SNE early exaggeration, NCVis init epochs) scale
together, so a short run keeps the same phase proportions.
"""

PERPLEXITY = 30.0
# t-SNE schedule shared by openTSNE and manifolds-rs (openTSNE's defaults): early exaggeration
# (factor 12) for the first iterations, then the plain phase. manifolds-rs ships 1000 iterations
# by default; it is aligned so equal budgets mean equal iterations.
TSNE_ITERATIONS = 750
TSNE_EARLY_EXAGGERATION_ITER = 250
TSNE_EARLY_EXAGGERATION = 12.0
N_NEIGHBORS = 15
MIN_DIST = 0.1

MIN_EPOCHS = 1


def scaled_epochs(default_epochs: int, budget: float) -> int:
    return max(MIN_EPOCHS, round(default_epochs * budget))


def tsne_schedule(budget: float) -> tuple[int, int]:
    """(early-exaggeration iterations, total iterations) at a budget, identical for both libraries."""
    return scaled_epochs(TSNE_EARLY_EXAGGERATION_ITER, budget), scaled_epochs(TSNE_ITERATIONS, budget)
