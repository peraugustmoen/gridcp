"""Scores for the grid detector.

All scores should be implemented as classes that follow the `ScoreModel` protocol defined
in `gridcp.typing`. This means they need to implement three methods.
See `_mean_cusum.py` for an example implementation.
"""

from gridcp.scores._mean_cusum import MeanCUSUM, MeanCUSUMState
from gridcp.scores._mean_cusum_unknown_variance import (
    MeanCUSUMUnknownVariance,
    MeanCUSUMUnknownVarianceState,
)
from gridcp.scores._exponential_family_glr import (
    ExponentialFamilyGLR,
    ExponentialFamilyGLRState,
)

__all__ = [
    "MeanCUSUM",
    "MeanCUSUMState",
    "MeanCUSUMUnknownVariance",
    "MeanCUSUMUnknownVarianceState",
    "ExponentialFamilyGLR",
    "ExponentialFamilyGLRState",
]
