from .core import init_state, update_func
from .calibration import mc_max_statistics
import numpy as np
from . import builtins


class OnlineChangepointDetector:
    """
    Online changepoint detection wrapper.

    This class maintains the state of an online changepoint detector, updating it
    sequentially as new observations arrive. Detection is driven by a user-specified
    feature map $h$, test statistic $f$, and penalty function.

    Parameters
    ----------
    p : int
        Dimension of data to be processed.
    h : callable
        Feature map applied to each new observation. Must accept a 1D NumPy array
        or float and return a 1D NumPy array. The output dimension of `h` must
        match the input dimension of `f`.
    f : callable
        Test statistic function. Must accept two numpy arrays of the same
        dimension as the output of `h` and return a float.
    penalty : callable
        Penalty function applied to candidate changepoint segment lengths.
        Must accept integers and return a float.
    penalty_constant : float
        Non-negative constant that scales  the penalty term used
        in the changepoint detection statistic.
    auxiliary_data : Any, optional
        Not implemented yet, but reserved for any additional data needed by
        e.g. `h`, `f`, or `penalty`.

    Notes
    -----
    The internal state is managed stored in the dictionary `self._state`,
    which contains the following keys:

    - `"t"` : current time index (int)
    - `"alarm"` : boolean flag indicating whether a changepoint has
       been detected (now or in the past)
    - `"maxx"` : maximum value of the detection statistic so far (float)
    - `"maxpos"` : time index at which `"maxx"` was attained (int)
    - `"penalty_constant"` : current penalty constant (float)
    - `"p"`, `"h"`, `"f"`, `"penalty"`, `"auxiliary_data"`
    """

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
        """
        Update the detector with a new observation.

        The internal state is updated in-place using `update_func`, and the
        method returns whether a changepoint alarm is raised after incorporating
        the new data point.

        Parameters
        ----------
        x_new : float or array_like
            New observation at the current time step. A scalar is treated as a
            one-dimensional array; otherwise it is converted to at least 1D
            (via `np.atleast_1d`) before passing through the feature map `h`.

        Returns
        -------
        bool
            `True` if the detector raises an alarm (i.e., a changepoint is
            detected) after this update.
        """
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

    @property
    def penalty_constant(self):
        return float(self._state["penalty_constant"])

    def calibrate_false_alarm(
        self, alpha, N, K, null_dist, null_args=(), null_kwargs={}, seed=42
    ):
        """
        Calibrate the penalty constant to target a given false alarm probability
        under a user-specified null distribution.

        This performs a Monte Carlo calibration using `mc_max_statistics` under a
        user-specified null distribution. The penalty constant is set to the
        $(1 - \\alpha)$-quantile of the simulated maximum statistics.

        Parameters
        ----------
        alpha : float
            Desired (per-experiment) false alarm probability in $(0, 1)$.
            The penalty constant is set so that the probability that the maximum
            statistic exceeds it under the null is approximately `alpha`.
        N : int
            Length (number of time steps) of each Monte Carlo simulation path.
        K : int
            Number of Monte Carlo replication paths to simulate.
        null_dist : callable
            Random variate generator for the null distribution of the data.
            Should support calls of the form `null_dist(*null_args, **null_kwargs)`
            and return samples in a format compatible with `h`.
        null_args : tuple, optional
            Positional arguments passed to `null_dist`.
        null_kwargs : dict, optional
            Keyword arguments passed to `null_dist`.
        seed : int, optional
            Random seed used by `mc_max_statistics` for reproducibility.

        Notes
        -----
        The original `penalty_constant` is ignored during calibration;after
        simulation, the detector's internal `"penalty_constant"` is overwritten with the
        calibrated value.
        """
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


def make_regression_change_detector(
    q,
    mode="McScan",
    penalty_constant=1.0,
) -> OnlineChangepointDetector:
    """
    Factory for a regression change detector based on McScan (Cho, Kley, Li 2025).
    Here, q is the number of regression coefficients.

    Parameters
    ----------
    penalty_constant : float
        Initial penalty constant (can be calibrated later with calibrate_false_alarm)
    Returns
    -------
    det : OnlineChangepointDetector
        Detector with pre-specified h, f, penalty.
    """

    h = builtins.h_regression_mcscan
    f = builtins.f_regression_mcscan
    penalty = builtins.penalty_regression_mcscan

    if mode == "direct":
        h = builtins.h_regression_direct
        f = builtins.f_regression_direct
        penalty = builtins.penalty_regression_direct

    det = OnlineChangepointDetector(
        p=q + 1,
        h=h,
        f=f,
        penalty=penalty,
        penalty_constant=penalty_constant,
    )
    return det
