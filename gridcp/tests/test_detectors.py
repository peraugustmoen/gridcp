import numpy as np
import numba as nb
import math
from time import perf_counter
import matplotlib.pyplot as plt
import gridcp

from gridcp import (
    OnlineChangepointDetector,
    make_univariate_mean_change_detector,
    make_univariate_variance_change_detector,
    make_univariate_mean_or_variance_change_detector,
    make_multivariate_mean_change_detector,
    make_multivariate_mean_or_covariance_change_detector,
    make_regression_change_detector,
)
from gridcp.utils import fastlog
from gridcp.calibration import draw_samples, mc_max_statistics
from gridcp import builtins
from gridcp.builtins import gen_gaussian_regression_obs


def test_univariate_mean_1():
    N = 1000

    ## check that
    np.random.seed(123)
    x = np.random.normal(loc=0, scale=1, size=N)

    detector = make_univariate_mean_change_detector(penalty_constant=5.0)
    for i in range(N):
        detector.update(x[i])

    assert not detector.alarm
    detector.update(1000000.0)
    assert detector.alarm

    ## check that cumulative sums are stored correctly
    np.random.seed(123)
    x = np.random.normal(loc=0, scale=1, size=N)

    detector = make_univariate_mean_change_detector(penalty_constant=5.0)
    for i in range(N):
        detector.update(x[i])
    cumsums = np.cumsum(x)
    d_grid = detector._state["grid_list"]
    d_sums = detector._state["sum_pre_list"]
    d_sum = detector._state["sum"]
    true_grid = gridcp.core.get_grid(N)
    true_grid = true_grid[::-1]
    true_ind = [N - g - 1 for g in true_grid]
    true_sums = cumsums[true_ind]

    assert len(d_sums) == len(true_sums)

    for s1, s2 in zip(d_sums, true_sums):
        assert np.isclose(s1, s2), f"Expected {s2}, got {s1}"

    S_true = np.sum(x)

    assert np.isclose(S_true, d_sum)
    ## check the CUSUM computation

    ff = detector._state["f"]

    for i in range(len(d_sums)):
        g = d_grid[i] + N + 1
        d_cumsum = d_sums[i]
        d_cusumval = ff(d_cumsum, d_sum - d_cumsum, N - g, N) + 1

        g_true = true_grid[i]

        assert g == g_true

        cumsum_true = true_sums[i]
        true_cusumval = gridcp.utils.CUSUM(S_true, cumsum_true, g_true, N)

        assert np.isclose(d_cusumval, true_cusumval)


def test_univariate_mean_2():
    ## check errenous inputs, etc!!
    pass


def test_multivariate_mean_1():
    N = 100
    p = 10

    ## check that
    np.random.seed(123)
    x = np.random.normal(loc=0, scale=1, size=(N, p))

    detector = make_multivariate_mean_change_detector(p=p, penalty_constant=5.0)
    for i in range(N):
        detector.update(x[i])

    assert not detector.alarm
    xnew = np.ones(p) * 100000.0
    detector.update(xnew)
    assert detector.alarm

    ## check that cumulative sums are stored correctly
    np.random.seed(123)
    x = np.random.normal(loc=0, scale=1, size=(N, p))

    detector = make_multivariate_mean_change_detector(p=p, penalty_constant=5.0)
    for i in range(N):
        detector.update(x[i])
    cumsums = np.cumsum(x, axis=0)

    d_grid = detector._state["grid_list"]
    d_sums = detector._state["sum_pre_list"]
    d_sum = detector._state["sum"]
    true_grid = gridcp.core.get_grid(N)
    true_grid = true_grid[::-1]
    true_ind = [N - g - 1 for g in true_grid]
    true_sums = cumsums[true_ind]

    assert len(d_sums) == len(true_sums)

    for s1, s2 in zip(d_sums, true_sums):
        assert np.isclose(s1, s2).all(), f"Expected {s2}, got {s1}"

    S_true = np.sum(x, axis=0)

    assert np.isclose(S_true, d_sum).all()
    ## check the CUSUM computation

    ff = detector._state["f"]

    for i in range(len(d_sums)):
        g = d_grid[i] + N + 1
        d_cumsum = d_sums[i]
        d_cusumval = ff(d_cumsum, d_sum - d_cumsum, N - g, N)

        g_true = true_grid[i]

        assert g == g_true

        cumsum_true = true_sums[i]
        true_cusumval = gridcp.utils.CUSUM(S_true, cumsum_true, g_true, N)
        true_cusumval = np.sum(true_cusumval) - p

        assert np.isclose(d_cusumval, true_cusumval)
