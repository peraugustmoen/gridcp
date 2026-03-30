"""Scores for the grid detector.

All scores should be implemented as classes that follow the `ScoreModel` protocol defined
in `gridcp.typing`. This means they need to implement three methods.
See `_mean_cusum.py` for an example implementation.
"""

from gridcp.typing import PenaltyType
from gridcp.scores._mean_cusum import MeanCUSUM
from gridcp.scores._mean_unknown_variance import MeanCUSUMUnknownVariance
from gridcp.scores._mean_or_variance import MeanOrVariance
from gridcp.scores._multivariate_mean_or_covariance import MultivariateMeanOrCovariance
from gridcp.scores._multivariate_mean_identity_cov import MultivariateMeanIdentityCov
from gridcp.scores._multivariate_mean_unknown_cov import MultivariateMeanUnknownCov
from gridcp.scores._regression_direct import RegressionDirect
from gridcp.scores._regression_mcscan import RegressionMcScan
from gridcp.scores._variance import Variance
from gridcp.scores._exponential_family_glr import ExponentialFamilyGLR


__all__ = [
    "PenaltyType",
    "MeanCUSUM",
    "MeanCUSUMUnknownVariance",
    "Variance",
    "MeanOrVariance",
    "MultivariateMeanIdentityCov",
    "MultivariateMeanUnknownCov",
    "MultivariateMeanOrCovariance",
    "RegressionMcScan",
    "RegressionDirect",
    "ExponentialFamilyGLR",
]
