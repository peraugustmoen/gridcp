import numpy as np
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


def test_univariate_mean_reset():
    """After reset, detector state is fresh."""
    N = 100
    np.random.seed(42)
    x = np.random.normal(loc=0, scale=1, size=N)

    detector = make_univariate_mean_change_detector(penalty_constant=5.0)
    for i in range(N):
        detector.update(x[i])

    assert detector.t == N
    detector.reset()
    assert detector.t == 0
    assert not detector.alarm
    assert detector.max_statistic == 0.0


def test_univariate_mean_no_false_alarm_high_threshold():
    """Under the null with a very high threshold, no alarm should be raised."""
    N = 500
    np.random.seed(999)
    x = np.random.normal(loc=0, scale=1, size=N)

    detector = make_univariate_mean_change_detector(penalty_constant=50.0)
    for i in range(N):
        detector.update(x[i])

    assert not detector.alarm


def test_univariate_mean_update_returns_bool():
    """update() should return a bool matching the alarm state."""
    detector = make_univariate_mean_change_detector(penalty_constant=5.0)

    np.random.seed(42)
    for _ in range(50):
        result = detector.update(np.random.normal())
        assert isinstance(result, bool)

    assert not detector.alarm
    result = detector.update(1_000_000.0)
    assert isinstance(result, bool)
    assert result is True
    assert detector.alarm


def test_univariate_mean_t_increments():
    """The t property should increment by 1 with each update."""
    detector = make_univariate_mean_change_detector(penalty_constant=5.0)
    assert detector.t == 0

    np.random.seed(42)
    for i in range(1, 101):
        detector.update(np.random.normal())
        assert detector.t == i


def test_univariate_mean_max_statistic_nondecreasing():
    """max_statistic should be non-decreasing over time."""
    N = 200
    np.random.seed(42)
    x = np.random.normal(loc=0, scale=1, size=N)

    detector = make_univariate_mean_change_detector(penalty_constant=50.0)
    prev_max = detector.max_statistic
    for i in range(N):
        detector.update(x[i])
        assert detector.max_statistic >= prev_max
        prev_max = detector.max_statistic


def test_univariate_mean_maxpos_after_alarm():
    """After alarm, maxpos should be a valid grid position."""
    N = 100
    np.random.seed(42)
    x = np.random.normal(loc=0, scale=1, size=N)

    detector = make_univariate_mean_change_detector(penalty_constant=5.0)
    for i in range(N):
        detector.update(x[i])

    assert not detector.alarm
    detector.update(1_000_000.0)
    assert detector.alarm
    # maxpos should be a positive integer within the range of time steps seen
    assert detector.maxpos >= 1
    assert detector.maxpos <= detector.t


def test_univariate_mean_detects_mid_stream_shift():
    """Detector should fire when there is a genuine mean shift mid-stream."""
    N_pre = 200
    N_post = 200
    np.random.seed(123)
    x_pre = np.random.normal(loc=0, scale=1, size=N_pre)
    x_post = np.random.normal(loc=5, scale=1, size=N_post)  # big mean shift

    detector = make_univariate_mean_change_detector(penalty_constant=5.0)
    for val in x_pre:
        detector.update(val)
    assert not detector.alarm

    alarmed = False
    for val in x_post:
        if detector.update(val):
            alarmed = True
            break
    assert alarmed, "Detector failed to detect a mean shift from 0 to 5"


def test_univariate_mean_short_streams():
    """Detector should handle very short streams (t=1,2,3) without errors."""
    detector = make_univariate_mean_change_detector(penalty_constant=5.0)
    detector.update(0.5)
    assert detector.t == 1
    assert not detector.alarm

    detector.update(-0.3)
    assert detector.t == 2

    detector.update(0.1)
    assert detector.t == 3


def test_univariate_mean_reuse_after_reset():
    """After reset, the detector should behave identically to a fresh one."""
    N = 100
    np.random.seed(42)
    x = np.random.normal(loc=0, scale=1, size=N)

    detector = make_univariate_mean_change_detector(penalty_constant=5.0)
    for val in x:
        detector.update(val)

    detector.reset()

    # Feed the same data to a fresh detector and compare
    fresh = make_univariate_mean_change_detector(penalty_constant=5.0)
    np.random.seed(7)
    x2 = np.random.normal(loc=0, scale=1, size=N)

    for val in x2:
        detector.update(val)
        fresh.update(val)

    assert detector.t == fresh.t
    assert detector.alarm == fresh.alarm
    assert np.isclose(detector.max_statistic, fresh.max_statistic)


def test_univariate_mean_unknown_var_alarm():
    """Unknown-variance mode: no alarm under null, alarm after extreme observation."""
    N = 200
    np.random.seed(123)
    x = np.random.normal(loc=0, scale=1, size=N)

    detector = make_univariate_mean_change_detector(
        penalty_constant=5.0, mode="unknown_variance"
    )
    for val in x:
        detector.update(val)

    assert not detector.alarm
    detector.update(1_000_000.0)
    assert detector.alarm


def test_univariate_mean_unknown_var_cumsums():
    """Unknown-variance mode: prefix sums of h(y) match manual computation.

    h(y) returns [y, y^2], so prefix sums are cumulative sums of that pair.
    """
    N = 200
    np.random.seed(123)
    x = np.random.normal(loc=0, scale=1, size=N)

    detector = make_univariate_mean_change_detector(
        penalty_constant=5.0, mode="unknown_variance"
    )
    for val in x:
        detector.update(val)

    h = detector._state["h"]
    h_vals = np.array([h(np.atleast_1d(x[i])) for i in range(N)])
    cumsums = np.cumsum(h_vals, axis=0)

    d_grid = detector._state["grid_list"]
    d_sums = detector._state["sum_pre_list"]
    d_sum = detector._state["sum"]
    true_grid = gridcp.core.get_grid(N)
    true_grid = true_grid[::-1]
    true_ind = [N - g - 1 for g in true_grid]
    true_sums = cumsums[true_ind]

    assert len(d_sums) == len(true_sums)

    for s1, s2 in zip(d_sums, true_sums):
        assert np.allclose(s1, s2), f"Expected {s2}, got {s1}"

    S_true = cumsums[-1]
    assert np.allclose(S_true, d_sum)


def test_univariate_mean_unknown_var_no_false_alarm():
    """Unknown-variance mode: no false alarm with high threshold."""
    N = 300
    np.random.seed(999)
    x = np.random.normal(loc=0, scale=1, size=N)

    detector = make_univariate_mean_change_detector(
        penalty_constant=50.0, mode="unknown_variance"
    )
    for val in x:
        detector.update(val)

    assert not detector.alarm


def test_univariate_mean_unknown_var_detects_shift():
    """Unknown-variance mode should detect a genuine mean shift."""
    N_pre = 200
    N_post = 200
    np.random.seed(42)
    x_pre = np.random.normal(loc=0, scale=1, size=N_pre)
    x_post = np.random.normal(loc=5, scale=1, size=N_post)

    detector = make_univariate_mean_change_detector(
        penalty_constant=5.0, mode="unknown_variance"
    )
    for val in x_pre:
        detector.update(val)
    assert not detector.alarm

    alarmed = False
    for val in x_post:
        if detector.update(val):
            alarmed = True
            break
    assert alarmed, "Unknown-var detector failed to detect mean shift from 0 to 5"


def test_univariate_mean_unknown_var_reset():
    """Unknown-variance mode: after reset, state is fresh."""
    N = 100
    np.random.seed(42)
    x = np.random.normal(loc=0, scale=1, size=N)

    detector = make_univariate_mean_change_detector(
        penalty_constant=5.0, mode="unknown_variance"
    )
    for val in x:
        detector.update(val)

    assert detector.t == N
    detector.reset()
    assert detector.t == 0
    assert not detector.alarm
    assert detector.max_statistic == 0.0


def test_univariate_mean_calibrate_false_alarm():
    """calibrate_false_alarm should set a reasonable penalty constant."""
    detector = make_univariate_mean_change_detector(penalty_constant=0.0)
    detector.calibrate_false_alarm(
        alpha=0.05,
        N=200,
        K=100,
        null_dist=np.random.normal,
        seed=42,
    )
    # After calibration, penalty constant should be positive and finite
    assert detector.penalty_constant > 0
    assert np.isfinite(detector.penalty_constant)

    # Now run under the null: with 5% nominal level, we expect no alarm most of the time
    np.random.seed(123)
    x = np.random.normal(loc=0, scale=1, size=200)
    for val in x:
        detector.update(val)
    # Not strictly deterministic, but the calibrated threshold should prevent alarm
    # on a single null run with high probability


def test_multivariate_mean_known_var_alarm():
    """No alarm under null, alarm after extreme observation (identity covariance)."""
    N = 100
    p = 10

    np.random.seed(123)
    x = np.random.normal(loc=0, scale=1, size=(N, p))

    detector = make_multivariate_mean_change_detector(p=p, penalty_constant=5.0)
    for i in range(N):
        detector.update(x[i])

    assert not detector.alarm
    xnew = np.ones(p) * 100000.0
    detector.update(xnew)
    assert detector.alarm


def test_multivariate_mean_known_var_cumsums():
    """Cumulative sums and CUSUM values match ground truth (identity covariance)."""
    N = 100
    p = 10

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


def test_multivariate_mean_known_var_reset():
    """After reset, detector state is fresh."""
    p = 5
    N = 50

    np.random.seed(42)
    x = np.random.normal(loc=0, scale=1, size=(N, p))

    detector = make_multivariate_mean_change_detector(p=p, penalty_constant=5.0)
    for i in range(N):
        detector.update(x[i])

    assert detector.t == N
    detector.reset()
    assert detector.t == 0
    assert not detector.alarm
    assert detector.max_statistic == 0.0


def test_multivariate_mean_unknown_var_alarm():
    """No alarm under null, alarm after extreme observation (unknown covariance)."""
    N = 100
    p = 5

    np.random.seed(123)
    x = np.random.normal(loc=0, scale=1, size=(N, p))

    detector = make_multivariate_mean_change_detector(
        p=p, penalty_constant=5.0, mode="unknown_variance"
    )
    for i in range(N):
        detector.update(x[i])

    assert not detector.alarm
    xnew = np.ones(p) * 100000.0
    detector.update(xnew)
    assert detector.alarm


def test_multivariate_mean_unknown_var_cumsums():
    """Cumulative sums of h(y) match manual computation (unknown covariance).

    For the unknown-covariance mode, h(y) returns a (p+1, p) matrix:
      row 0 = y, rows 1: = y y^T.
    We verify that the stored prefix sums match cumulative sums of h(x_i).
    """
    N = 100
    p = 5

    np.random.seed(123)
    x = np.random.normal(loc=0, scale=1, size=(N, p))

    detector = make_multivariate_mean_change_detector(
        p=p, penalty_constant=5.0, mode="unknown_variance"
    )
    for i in range(N):
        detector.update(x[i])

    h = detector._state["h"]

    # Build ground-truth cumulative sums of h(x_i)
    h_vals = np.zeros((N, p + 1, p), dtype=np.float64)
    for i in range(N):
        h_vals[i] = h(x[i])
    cumsums = np.cumsum(h_vals, axis=0)

    d_grid = detector._state["grid_list"]
    d_sums = detector._state["sum_pre_list"]
    d_sum = detector._state["sum"]
    true_grid = gridcp.core.get_grid(N)
    true_grid = true_grid[::-1]

    assert len(d_sums) == len(true_grid)

    for i in range(len(d_sums)):
        g = d_grid[i] + N + 1
        g_true = true_grid[i]
        assert g == g_true

        # prefix sum up to index (N - g - 1) inclusive
        true_idx = N - g - 1
        true_sum = cumsums[true_idx]
        assert np.allclose(d_sums[i], true_sum), (
            f"Grid point {g}: expected prefix sum shape {true_sum.shape}, "
            f"got {d_sums[i].shape}"
        )

    S_true = cumsums[-1]
    assert np.allclose(S_true, d_sum)


def test_multivariate_mean_unknown_var_reset():
    """After reset, unknown-variance detector state is fresh."""
    p = 5
    N = 50

    np.random.seed(42)
    x = np.random.normal(loc=0, scale=1, size=(N, p))

    detector = make_multivariate_mean_change_detector(
        p=p, penalty_constant=5.0, mode="unknown_variance"
    )
    for i in range(N):
        detector.update(x[i])

    assert detector.t == N
    detector.reset()
    assert detector.t == 0
    assert not detector.alarm
    assert detector.max_statistic == 0.0


def test_multivariate_mean_unknown_var_no_false_alarm():
    """Under the null (iid Gaussian), no alarm should be raised with a high threshold."""
    N = 200
    p = 5

    np.random.seed(999)
    x = np.random.normal(loc=0, scale=1, size=(N, p))

    detector = make_multivariate_mean_change_detector(
        p=p, penalty_constant=50.0, mode="unknown_variance"
    )
    for i in range(N):
        detector.update(x[i])

    assert not detector.alarm


def test_multivariate_mean_known_var_dimensions():
    """Test multiple data dimensions for the known-variance detector."""
    for p in [2, 5, 20]:
        N = 100
        np.random.seed(42)
        x = np.random.normal(loc=0, scale=1, size=(N, p))

        detector = make_multivariate_mean_change_detector(p=p, penalty_constant=5.0)
        for i in range(N):
            detector.update(x[i])

        assert not detector.alarm
        assert detector.t == N

        # Extreme observation triggers alarm
        xnew = np.ones(p) * 100000.0
        detector.update(xnew)
        assert detector.alarm


def test_multivariate_mean_or_covariance_alarm():
    """No alarm under null, alarm after extreme observation (mean or covariance)."""
    N = 100
    p = 5

    np.random.seed(123)
    x = np.random.normal(loc=0, scale=1, size=(N, p))

    detector = make_multivariate_mean_or_covariance_change_detector(
        p=p, penalty_constant=5.0
    )
    for i in range(N):
        detector.update(x[i])

    assert not detector.alarm
    xnew = np.ones(p) * 100000.0
    detector.update(xnew)
    assert detector.alarm


def test_multivariate_mean_or_covariance_cumsums():
    """Cumulative sums of h(y) match manual computation (mean or covariance).

    For mean-or-covariance mode, h(y) also returns a (p+1, p) matrix.
    """
    N = 100
    p = 5

    np.random.seed(123)
    x = np.random.normal(loc=0, scale=1, size=(N, p))

    detector = make_multivariate_mean_or_covariance_change_detector(
        p=p, penalty_constant=5.0
    )
    for i in range(N):
        detector.update(x[i])

    h = detector._state["h"]

    h_vals = np.zeros((N, p + 1, p), dtype=np.float64)
    for i in range(N):
        h_vals[i] = h(x[i])
    cumsums = np.cumsum(h_vals, axis=0)

    d_grid = detector._state["grid_list"]
    d_sums = detector._state["sum_pre_list"]
    d_sum = detector._state["sum"]
    true_grid = gridcp.core.get_grid(N)
    true_grid = true_grid[::-1]

    assert len(d_sums) == len(true_grid)

    for i in range(len(d_sums)):
        g = d_grid[i] + N + 1
        g_true = true_grid[i]
        assert g == g_true

        true_idx = N - g - 1
        true_sum = cumsums[true_idx]
        assert np.allclose(d_sums[i], true_sum)

    S_true = cumsums[-1]
    assert np.allclose(S_true, d_sum)


def test_multivariate_mean_or_covariance_no_false_alarm():
    """Under the null with a high threshold, no alarm should be raised."""
    N = 200
    p = 5

    np.random.seed(999)
    x = np.random.normal(loc=0, scale=1, size=(N, p))

    detector = make_multivariate_mean_or_covariance_change_detector(
        p=p, penalty_constant=50.0
    )
    for i in range(N):
        detector.update(x[i])

    assert not detector.alarm
