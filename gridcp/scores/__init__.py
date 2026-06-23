"""Scores for the grid detector.

All scores should be implemented as classes that follow the `ScoreModel` protocol defined
in `gridcp.typing`. This means they need to implement three methods.
See `_cusum.py` for an example implementation.
"""

from gridcp.scores._cusum import CUSUM
from gridcp.scores._exponential_family_glr import ExponentialFamilyGLR
from gridcp.scores._gaussian_mean import GaussianMean
from gridcp.scores._gaussian_mean_full_covariance import GaussianMeanFullCovariance
from gridcp.scores._gaussian_mean_or_covariance import GaussianMeanOrCovariance
from gridcp.scores._gaussian_mean_or_variance import GaussianMeanOrVariance
from gridcp.scores._gaussian_variance import GaussianVariance
from gridcp.scores._npfocus import NPFOCuS
from gridcp.scores._regression_mcscan import RegressionMcScan
from gridcp.scores._regression_wald import RegressionWald


BUILTIN_SCORE_TYPES = (
    CUSUM,
    GaussianMean,
    GaussianVariance,
    GaussianMeanOrVariance,
    GaussianMeanFullCovariance,
    GaussianMeanOrCovariance,
    RegressionMcScan,
    RegressionWald,
    ExponentialFamilyGLR,
    NPFOCuS,
)


__all__ = [
    "CUSUM",
    "GaussianMean",
    "GaussianVariance",
    "GaussianMeanOrVariance",
    "GaussianMeanFullCovariance",
    "GaussianMeanOrCovariance",
    "RegressionMcScan",
    "RegressionWald",
    "ExponentialFamilyGLR",
    "NPFOCuS",
    "BUILTIN_SCORE_TYPES",
]
