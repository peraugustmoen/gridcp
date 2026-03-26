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
from gridcp.scores._mean_or_variance_lr import MeanOrVarianceLR, MeanOrVarianceLRState
from gridcp.scores._multivariate_mean_and_covariance_lr import (
    MultivariateMeanAndCovarianceLR,
    MultivariateMeanAndCovarianceLRState,
)
from gridcp.scores._multivariate_mean_identity_cov_lr import (
    MultivariateMeanIdentityCovLR,
    MultivariateMeanIdentityCovLRState,
)
from gridcp.scores._multivariate_mean_unknown_cov_lr import (
    MultivariateMeanUnknownCovLR,
    MultivariateMeanUnknownCovLRState,
)
from gridcp.scores._regression_direct import RegressionDirect, RegressionDirectState
from gridcp.scores._regression_mcscan import RegressionMcScan, RegressionMcScanState
from gridcp.scores._variance_lr import VarianceLR, VarianceLRState

__all__ = [
    "MeanCUSUM",
    "MeanCUSUMState",
    "MeanCUSUMUnknownVariance",
    "MeanCUSUMUnknownVarianceState",
    "VarianceLR",
    "VarianceLRState",
    "MeanOrVarianceLR",
    "MeanOrVarianceLRState",
    "MultivariateMeanIdentityCovLR",
    "MultivariateMeanIdentityCovLRState",
    "MultivariateMeanUnknownCovLR",
    "MultivariateMeanUnknownCovLRState",
    "MultivariateMeanAndCovarianceLR",
    "MultivariateMeanAndCovarianceLRState",
    "RegressionMcScan",
    "RegressionMcScanState",
    "RegressionDirect",
    "RegressionDirectState",
]
