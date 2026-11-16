"""Hyperparameters shared across implementations and the iteration-budget scaling.

A budget multiplier scales an algorithm's own default number of optimisation epochs (×1 = the
library default). Multi-phase schedules (t-SNE early exaggeration, NCVis init epochs) scale
together, so a short run keeps the same phase proportions.
"""

PERPLEXITY = 30.0
N_NEIGHBORS = 15
MIN_DIST = 0.1

MIN_EPOCHS = 1


def scaled_epochs(default_epochs: int, budget: float) -> int:
    return max(MIN_EPOCHS, round(default_epochs * budget))
