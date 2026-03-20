"""Scores for the grid detector.

All scores should be implemented as classes that follow the `GridScore` protocol defined
in `gridcp.new_api.typing`. This means they need to implement three methods.
See `_mean_cusum.py` for an example implementation.
"""

from gridcp.new_api.scores._mean_cusum import MeanCUSUM, MeanCUSUMState

__all__ = [
    "MeanCUSUM",
    "MeanCUSUMState",
]
