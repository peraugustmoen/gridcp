from .core import init_state, update_func
from .calibration import mc_max_statistics
import numpy as np
from . import builtins


class OnlineChangepointDetector:
    ## f accepts numpy arrays as input and outputs a float
    ## h accepts a numpy array and outputs a numpy array (can be identity)
    ## dimension v of output of h must match input dimensions of f
    ## penalty accepts integers and outputs a float

    def __init__(self, p, h, f, penalty, penalty_constant, auxiliary_data=None):
        self._state = init_state(
            p=p,
            h=h,
            f=f,
            penalty=penalty,
            penalty_constant=penalty_constant,
            auxiliary_data=auxiliary_data,
        )

    def update(self, x_new):
        # input may be float or 1D array, but we will convert to 1D array for consistency (hence the np.atleast_1d)
        update_func(x_new, self._state)
        return bool(self._state["alarm"])

    @property
    def t(self):
        return int(self._state["t"])

    @property
    def alarm(self):
        return bool(self._state["alarm"])

    @property
    def max_statistic(self):
        return float(self._state["maxx"])

    @property
    def maxpos(self):
        return int(self._state["maxpos"])

    def calibrate_false_alarm(
        self, alpha, N, K, null_dist, null_args=(), null_kwargs={}, seed=42
    ):
        mc_samp = mc_max_statistics(
            N=N,
            K=K,
            p=self._state["p"],
            h=self._state["h"],
            f=self._state["f"],
            penalty=self._state["penalty"],
            null_dist=null_dist,
            penalty_constant=0.0,
            null_dist_args=null_args,
            null_dist_kwargs=null_kwargs,
            auxiliary_data=None,
            seed=seed,
        )
        self._state["penalty_constant"] = float(np.quantile(mc_samp, 1 - alpha))

    def reset(self):
        p = self._state["p"]
        h = self._state["h"]
        f = self._state["f"]
        penalty = self._state["penalty"]
        penalty_constant = self._state["penalty_constant"]
        auxiliary_data = self._state["auxiliary_data"]
        self._state = init_state(
            p=p,
            h=h,
            f=f,
            penalty=penalty,
            penalty_constant=penalty_constant,
            auxiliary_data=auxiliary_data,
        )


#### Factory functions for built-in detectors ####
def make_univariate_mean_change_detector(
    penalty_constant=1.0, mode="known_variance"
) -> OnlineChangepointDetector:
    """
    Factory for a univariate Gaussian mean-change detector.

    Parameters
    ----------
    penalty_constant : float
        Initial penalty constant (can be calibrated later with calibrate_false_alarm)
    Returns
    -------
    det : OnlineChangepointDetector
        Detector with pre-specified h, f, penalty.
    """
    h = builtins.h_univariate_mean_known_var_LR
    f = builtins.f_univariate_mean_known_var_LR
    penalty = builtins.penalty_univariate_mean_known_var_LR
    if mode != "known_variance":
        h = builtins.h_univariate_mean_unknown_var_LR
        f = builtins.f_univariate_mean_unknown_var_LR
        penalty = builtins.penalty_univariate_mean_unknown_var_LR

    det = OnlineChangepointDetector(
        p=1,
        h=h,
        f=f,
        penalty=penalty,
        penalty_constant=penalty_constant,
    )
    return det


def make_univariate_variance_change_detector(
    penalty_constant=1.0,
) -> OnlineChangepointDetector:
    """
    Factory for a univariate Gaussian variance-change detector.

    Parameters
    ----------
    penalty_constant : float
        Initial penalty constant (can be calibrated later with calibrate_false_alarm)
    Returns
    -------
    det : OnlineChangepointDetector
        Detector with pre-specified h, f, penalty.
    """
    h = builtins.h_univariate_variance_LR
    f = builtins.f_univariate_variance_LR
    penalty = builtins.penalty_univariate_variance_LR

    det = OnlineChangepointDetector(
        p=1,
        h=h,
        f=f,
        penalty=penalty,
        penalty_constant=penalty_constant,
    )
    return det


def make_univariate_mean_or_variance_change_detector(
    penalty_constant=1.0,
) -> OnlineChangepointDetector:
    """
    Factory for a univariate Gaussian mean-or-variance change detector.

    Parameters
    ----------
    penalty_constant : float
        Initial penalty constant (can be calibrated later with calibrate_false_alarm)
    Returns
    -------
    det : OnlineChangepointDetector
        Detector with pre-specified h, f, penalty.
    """

    det = OnlineChangepointDetector(
        p=1,
        h=builtins.h_univariate_mean_or_variance_LR,
        f=builtins.f_univariate_mean_or_variance_LR,
        penalty=builtins.penalty_univariate_mean_or_variance_LR,
        penalty_constant=penalty_constant,
    )
    return det


def make_multivariate_mean_change_detector(
    p, penalty_constant=1.0, mode="known_variance"
) -> OnlineChangepointDetector:
    """
    Factory for a univariate Gaussian mean-change detector.

    Parameters
    ----------
    penalty_constant : float
        Initial penalty constant (can be calibrated later with calibrate_false_alarm)
    Returns
    -------
    det : OnlineChangepointDetector
        Detector with pre-specified h, f, penalty.
    """
    h = builtins.h_multivariate_mean_id_cov_LR
    f = builtins.f_multivariate_mean_id_cov_LR
    penalty = builtins.penalty_multivariate_mean_id_cov_LR
    if mode == "unknown_variance":
        h = builtins.h_multivariate_mean_unknown_cov_LR
        f = builtins.f_multivariate_mean_unknown_cov_LR
        penalty = builtins.penalty_multivariate_mean_unknown_cov_LR

    det = OnlineChangepointDetector(
        p=p,
        h=h,
        f=f,
        penalty=penalty,
        penalty_constant=penalty_constant,
    )
    return det


def make_multivariate_mean_or_covariance_change_detector(
    p,
    penalty_constant=1.0,
) -> OnlineChangepointDetector:
    """
    Factory for a multivariate Gaussian mean-or-covariance change detector.

    Parameters
    ----------
    penalty_constant : float
        Initial penalty constant (can be calibrated later with calibrate_false_alarm)
    Returns
    -------
    det : OnlineChangepointDetector
        Detector with pre-specified h, f, penalty.
    """

    det = OnlineChangepointDetector(
        p=p,
        h=builtins.h_multivariate_mean_and_covariance_LR,
        f=builtins.f_multivariate_mean_and_covariance_LR,
        penalty=builtins.penalty_multivariate_mean_and_covariance_LR,
        penalty_constant=penalty_constant,
    )
    return det
