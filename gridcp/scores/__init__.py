"""Scores for the grid detector.

All scores should be implemented as classes that follow the `ScoreModel` protocol defined
in `gridcp.typing`. This means they need to implement three methods.
See `_mean_cusum.py` for an example implementation.
"""

from gridcp.typing import PenaltyType
from gridcp.scores._mean_cusum import MeanCUSUM, MeanCUSUMState
from gridcp.scores._mean_unknown_variance import (
    MeanCUSUMUnknownVariance,
    MeanCUSUMUnknownVarianceState,
)
from gridcp.scores._mean_or_variance import MeanOrVariance, MeanOrVarianceState
from gridcp.scores._multivariate_mean_or_covariance import (
    MultivariateMeanOrCovariance,
    MultivariateMeanOrCovarianceState,
)
from gridcp.scores._multivariate_mean_identity_cov import (
    MultivariateMeanIdentityCov,
    MultivariateMeanIdentityCovState,
)
from gridcp.scores._multivariate_mean_unknown_cov import (
    MultivariateMeanUnknownCov,
    MultivariateMeanUnknownCovState,
)
from gridcp.scores._regression_direct import RegressionDirect, RegressionDirectState
from gridcp.scores._regression_mcscan import RegressionMcScan, RegressionMcScanState
from gridcp.scores._variance import Variance, VarianceState

__all__ = [
    "PenaltyType",
    "MeanCUSUM",
    "MeanCUSUMState",
    "MeanCUSUMUnknownVariance",
    "MeanCUSUMUnknownVarianceState",
    "Variance",
    "VarianceState",
    "MeanOrVariance",
    "MeanOrVarianceState",
    "MultivariateMeanIdentityCov",
    "MultivariateMeanIdentityCovState",
    "MultivariateMeanUnknownCov",
    "MultivariateMeanUnknownCovState",
    "MultivariateMeanOrCovariance",
    "MultivariateMeanOrCovarianceState",
    "RegressionMcScan",
    "RegressionMcScanState",
    "RegressionDirect",
    "RegressionDirectState",
]
