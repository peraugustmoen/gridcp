import numpy as np
import pytest

from gridcp.detector import GridDetector
from gridcp.scores import (
    MeanOrVariance,
    MultivariateMeanIdentityCov,
    MultivariateMeanOrCovariance,
    MultivariateMeanUnknownCov,
    RegressionDirect,
    RegressionMcScan,
    Variance,
)
from gridcp.utils import get_changeloc_grid


def _run_stream(detector: GridDetector, x: np.ndarray):
    state = detector.init_state()
    outputs = []
    for row in x:
        state, out = detector.update(state, row)
        outputs.append(out)
    return state, outputs


def test_multivariate_mean_known_var_alarm():
    """No alarm under null, alarm after extreme observation."""
    n = 100
    p = 10

    rng = np.random.default_rng(123)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanIdentityCov(n_features=p),
        threshold=np.array([5.0, 5.0], dtype=np.float64),
    )
    state, outputs = _run_stream(detector, x)

    assert not any(out["alarm"] for out in outputs)

    state, out = detector.update(state, np.ones(p, dtype=np.float64) * 100000.0)
    assert out["alarm"]


def test_multivariate_mean_known_var_cumsums_and_scores():
    """Prefix sums and raw LR scores match direct formula."""
    n = 100
    p = 10

    rng = np.random.default_rng(123)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanIdentityCov(n_features=p),
        threshold=np.array([100.0, 100.0], dtype=np.float64),
    )
    state, _ = _run_stream(detector, x)

    cumsums = np.cumsum(x, axis=0)
    true_grid = get_changeloc_grid(n)

    assert np.array_equal(np.asarray(state.grid, dtype=np.int64), true_grid)

    for i, n1 in enumerate(state.grid):
        expected_sum = cumsums[n1 - 1]
        actual_sum = state.previous_score_states[i].sum
        assert np.allclose(actual_sum, expected_sum)

        n2 = n - n1
        total_sum = state.current_score_state.sum
        mean1 = expected_sum / n1
        mean2 = (total_sum - expected_sum) / n2
        diff = mean1 - mean2
        lr = (n1 * n2 / n) * float(np.dot(diff, diff))
        expected_raw = lr - p

        scores = detector.score.compute_penalized_scores(
            state.current_score_state,
            state.previous_score_states,
        )[i]

        # Column 0: sparse statistic (max of squared diffs).
        lr_sparse = (n1 * n2 / n) * float(np.max(diff * diff))
        penalty_sparse = np.log(n) + np.log(p)
        assert np.isclose(scores[0], (lr_sparse - 1) / penalty_sparse)

        # Column 1: dense statistic (sum of squared diffs).
        penalty_dense = np.sqrt(p * np.log(n)) + np.log(n)
        assert np.isclose(scores[1], expected_raw / penalty_dense)


def test_multivariate_mean_known_var_reset_semantics():
    """Reinitializing state gives a fresh detector run."""
    p = 5
    n = 50

    rng = np.random.default_rng(42)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanIdentityCov(n_features=p),
        threshold=np.array([5.0, 5.0], dtype=np.float64),
    )
    state, _ = _run_stream(detector, x)
    assert state.n_samples == n

    fresh_state = detector.init_state()
    assert fresh_state.n_samples == 0
    assert fresh_state.grid == []


def test_multivariate_mean_unknown_var_alarm():
    """No alarm under null, alarm after extreme observations.

    The guard requires t >= 2*p+2 (more conservative than strict invertibility).
    After feeding 100 null observations (t=100 >> 2*p+2=12 for p=5), a few
    extreme post-change observations suffice to trigger an alarm.
    """
    n = 100
    p = 5

    rng = np.random.default_rng(123)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanUnknownCov(n_features=p),
        threshold=50.0,
    )
    state, outputs = _run_stream(detector, x)

    assert not any(out["alarm"] for out in outputs)

    # Feed a small number of extreme observations.  Once t >= 2*p+2 candidates
    # are evaluated; the alarm fires quickly under large signal.
    extreme_obs = np.ones(p, dtype=np.float64) * 100000.0
    alarmed = False
    for _ in range(p + 2):
        state, out = detector.update(state, extreme_obs)
        if out["alarm"]:
            alarmed = True
            break
    assert alarmed


def test_multivariate_mean_unknown_var_cumsums():
    """Stored prefix sums and second moments match manual cumulative sums."""
    n = 100
    p = 5

    rng = np.random.default_rng(123)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanUnknownCov(n_features=p),
        threshold=100.0,
    )
    state, _ = _run_stream(detector, x)

    cumsums = np.cumsum(x, axis=0)
    cum_outers = np.cumsum(
        np.stack([np.outer(x[i], x[i]) for i in range(n)], axis=0),
        axis=0,
    )

    true_grid = get_changeloc_grid(n)
    assert np.array_equal(np.asarray(state.grid, dtype=np.int64), true_grid)

    for i, n1 in enumerate(state.grid):
        expected_sum = cumsums[n1 - 1]
        expected_outer = cum_outers[n1 - 1]
        actual = state.previous_score_states[i]
        assert np.allclose(actual.sum, expected_sum)
        assert np.allclose(actual.sum_outer, expected_outer)

    assert np.allclose(state.current_score_state.sum, cumsums[-1])
    assert np.allclose(state.current_score_state.sum_outer, cum_outers[-1])


def test_multivariate_mean_unknown_var_no_false_alarm():
    """Under null with high threshold, no alarm should occur."""
    n = 200
    p = 5

    rng = np.random.default_rng(999)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanUnknownCov(n_features=p),
        threshold=50.0,
    )
    _, outputs = _run_stream(detector, x)

    assert not any(out["alarm"] for out in outputs)


def test_multivariate_mean_or_covariance_alarm():
    """No alarm under null, alarm after extreme observation."""
    n = 100
    p = 5

    rng = np.random.default_rng(123)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanOrCovariance(n_features=p),
        threshold=5.0,
    )
    state, outputs = _run_stream(detector, x)

    assert not any(out["alarm"] for out in outputs)

    state, out = detector.update(state, np.ones(p, dtype=np.float64) * 100000.0)
    assert out["alarm"]


def test_multivariate_mean_or_covariance_cumsums():
    """Stored prefix sums and second moments match manual cumulative sums."""
    n = 100
    p = 5

    rng = np.random.default_rng(123)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanOrCovariance(n_features=p),
        threshold=100.0,
    )
    state, _ = _run_stream(detector, x)

    cumsums = np.cumsum(x, axis=0)
    cum_outers = np.cumsum(
        np.stack([np.outer(x[i], x[i]) for i in range(n)], axis=0),
        axis=0,
    )

    true_grid = get_changeloc_grid(n)
    assert np.array_equal(np.asarray(state.grid, dtype=np.int64), true_grid)

    for i, n1 in enumerate(state.grid):
        expected_sum = cumsums[n1 - 1]
        expected_outer = cum_outers[n1 - 1]
        actual = state.previous_score_states[i]
        assert np.allclose(actual.sum, expected_sum)
        assert np.allclose(actual.sum_outer, expected_outer)

    assert np.allclose(state.current_score_state.sum, cumsums[-1])
    assert np.allclose(state.current_score_state.sum_outer, cum_outers[-1])


def test_multivariate_mean_or_covariance_no_false_alarm():
    """Under null with high threshold, no alarm should occur."""
    n = 200
    p = 5

    rng = np.random.default_rng(999)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanOrCovariance(n_features=p),
        threshold=50.0,
    )
    _, outputs = _run_stream(detector, x)

    assert not any(out["alarm"] for out in outputs)


def test_multivariate_identity_cov_broadcasts_scalar_threshold_silently():
    """A scalar threshold is broadcast for multivariate scores."""
    p = 4
    detector = GridDetector(
        score=MultivariateMeanIdentityCov(n_features=p),
        threshold=1.0,
    )
    state = detector.init_state()

    state, _ = detector.update(state, np.zeros(p, dtype=np.float64))
    state, out = detector.update(state, np.zeros(p, dtype=np.float64))
    state, _ = detector.update(state, np.zeros(p, dtype=np.float64))

    assert np.asarray(out["max_score"]).shape == (2,)
    assert np.asarray(out["max_split_point"]).shape == (2,)


def test_multivariate_identity_cov_rejects_wrong_threshold_length():
    """Threshold vector length must match the number of score components.

    With the dimension contract in place, the mismatch is caught at construction
    time (not deferred to update), so GridDetector.__post_init__ should raise.
    """
    p = 4
    with pytest.raises(ValueError, match="n_scores|threshold"):
        GridDetector(
            score=MultivariateMeanIdentityCov(n_features=p),
            threshold=np.array([1.0], dtype=np.float64),
        )


def test_multivariate_identity_cov_accepts_matching_threshold_length():
    """Strict mode accepts a threshold vector that matches score dimension."""
    p = 4
    detector = GridDetector(
        score=MultivariateMeanIdentityCov(n_features=p),
        threshold=np.array([1.0, 1.0], dtype=np.float64),
    )
    state = detector.init_state()

    state, out = detector.update(state, np.zeros(p, dtype=np.float64))
    assert np.asarray(out["max_split_point"]).shape == (2,)

    state, out = detector.update(state, np.zeros(p, dtype=np.float64))
    assert np.asarray(out["max_score"]).shape == (2,)
    assert np.asarray(out["max_split_point"]).shape == (2,)


# ---------------------------------------------------------------------------
# Sample-size guard boundary tests
# ---------------------------------------------------------------------------


def test_mean_or_cov_guard_at_boundary():
    """Score is 0 when n1 <= 2*p; non-zero when both n1 > 2*p and n2 > 2*p.

    The conservative guard requires each segment to have at least 2*p+1
    observations so that the covariance estimate is sufficiently stable for
    the null LR distribution to be close to chi-squared.
    """
    p = 4
    rng = np.random.default_rng(0)
    score = MultivariateMeanOrCovariance(n_features=p)

    # t = 2*(2*p+1) so that n1=2*p+1 and n2=2*p+1 is the minimal valid split.
    t = 2 * (2 * p + 1)
    x = rng.normal(size=(t, p))

    total = score.init_state()
    for xi in x:
        total = score.update(total, xi)

    # n1 = 2*p: just at the guard boundary, blocked.
    grid_n1_2p = score.init_state()
    for xi in x[: 2 * p]:
        grid_n1_2p = score.update(grid_n1_2p, xi)

    # n1 = 2*p+1: minimal valid split; n2 = 2*p+1 > 2*p as well.
    grid_n1_2p1 = score.init_state()
    for xi in x[: 2 * p + 1]:
        grid_n1_2p1 = score.update(grid_n1_2p1, xi)

    scores_below = score._compute_centered_scores(total, [grid_n1_2p])
    scores_at = score._compute_centered_scores(total, [grid_n1_2p1])

    assert scores_below[0] == 0.0
    assert scores_at[0] != 0.0


def test_mean_unknown_cov_guard_at_boundary():
    """Score requires t >= 2*p+2; individual segment sizes do not matter.

    The pooled within-group scatter (sigma_alt) sums scatter from both
    segments, so its rank = min(t-2, p). The conservative guard t >= 2*p+2
    ensures numerical stability of the chi-squared approximation.  Crucially,
    the individual segment sizes n1 and n2 play no role: a split with n2 = 1
    is valid once t >= 2*p+2 (the pooled scatter uses both segments).
    """
    p = 4
    rng = np.random.default_rng(0)
    score = MultivariateMeanUnknownCov(n_features=p)

    # t = 2*p+1: just below the guard, all scores must be 0.
    x_short = rng.normal(size=(2 * p + 1, p))
    total_short = score.init_state()
    for xi in x_short:
        total_short = score.update(total_short, xi)

    grid_n1_1_short = score.init_state()
    grid_n1_1_short = score.update(grid_n1_1_short, x_short[0])

    assert score._compute_centered_scores(total_short, [grid_n1_1_short])[0] == 0.0

    # t = 2*p+2, n1 = 2*p+1, n2 = 1:
    # S_W2 = 0 (single obs), sigma_alt = S_W1/t, rank = t-2 = 2*p >= p. Valid.
    x_enough = rng.normal(size=(2 * p + 2, p))
    total_enough = score.init_state()
    for xi in x_enough:
        total_enough = score.update(total_enough, xi)

    grid_n1_2p1 = score.init_state()
    for xi in x_enough[: 2 * p + 1]:
        grid_n1_2p1 = score.update(grid_n1_2p1, xi)

    assert score._compute_centered_scores(total_enough, [grid_n1_2p1])[0] != 0.0


def test_regression_direct_guard_at_boundary():
    """Score is 0 when n1 < q; non-zero when n1 >= q (Gram matrix full rank).

    The uncentered Gram matrix xx_pre = sum x_i x_i^T has rank min(n1, q)
    for observations in general position. No mean-centering means full rank
    is reached at n1 = q (one fewer than a centered covariance would need).
    """
    q = 4
    rng = np.random.default_rng(0)
    score = RegressionDirect(n_regressors=q)

    # t = 2*q so both segments simultaneously sit at the boundary.
    t = 2 * q
    x = rng.normal(size=(t, q + 1))  # columns: [y, x_1, ..., x_q]

    total = score.init_state()
    for xi in x:
        total = score.update(total, xi)

    # n1 = q-1: Gram matrix is rank q-1 < q, blocked.
    grid_n1_qm1 = score.init_state()
    for xi in x[: q - 1]:
        grid_n1_qm1 = score.update(grid_n1_qm1, xi)

    # n1 = q: Gram matrix is rank q, minimal valid split (n2 = q >= q).
    grid_n1_q = score.init_state()
    for xi in x[:q]:
        grid_n1_q = score.update(grid_n1_q, xi)

    scores_below = score._compute_centered_scores(total, [grid_n1_qm1])
    scores_at = score._compute_centered_scores(total, [grid_n1_q])

    assert scores_below[0] == 0.0
    assert scores_at[0] != 0.0


def test_no_runtime_warnings_mean_or_covariance():
    """No RuntimeWarning is raised during a long null stream (p=20).

    With correct guards slogdet is never called on a singular matrix,
    so no divide-by-zero warning can escape.
    """
    import warnings

    p = 20
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, p))
    detector = GridDetector(
        score=MultivariateMeanOrCovariance(n_features=p), threshold=100.0
    )
    state = detector.init_state()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for xi in x:
            state, _ = detector.update(state, xi)


def test_no_runtime_warnings_mean_unknown_cov():
    """No RuntimeWarning is raised during a long null stream (p=20).

    With the conservative guard t >= 2*p+2 and no inner segment-size
    guard, slogdet is only called when the pooled scatter matrix is
    genuinely full rank.
    """
    import warnings

    p = 20
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, p))
    detector = GridDetector(
        score=MultivariateMeanUnknownCov(n_features=p), threshold=100.0
    )
    state = detector.init_state()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for xi in x:
            state, _ = detector.update(state, xi)


# ---------------------------------------------------------------------------
# Variance parity tests
# ---------------------------------------------------------------------------


def test_variance_alarm():
    """No alarm under null, alarm after extreme observation."""
    n = 100
    p = 3
    rng = np.random.default_rng(7)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(score=Variance(n_features=p), threshold=20.0)
    state, outputs = _run_stream(detector, x)
    assert not any(out["alarm"] for out in outputs)

    # An extreme observation in one coordinate should trip the variance alarm.
    state, out = detector.update(state, np.array([1000.0] + [0.0] * (p - 1)))
    assert out["alarm"]


def test_variance_scores_match_formula():
    """_compute_centered_scores matches the per-feature LR formula."""
    n = 60
    p = 2
    rng = np.random.default_rng(42)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    score = Variance(n_features=p)
    total = score.init_state()
    for xi in x:
        total = score.update(total, xi)

    # Build a single grid state at n1 = 20.
    n1 = 20
    grid_state = score.init_state()
    for xi in x[:n1]:
        grid_state = score.update(grid_state, xi)

    result = score._compute_centered_scores(total, [grid_state])

    t = n
    n2 = t - n1
    sumsq = np.cumsum(x**2, axis=0)
    sumsq1 = sumsq[n1 - 1]
    sumsq_tot = sumsq[t - 1]
    sumsq2 = sumsq_tot - sumsq1

    expected = -1.0e300
    for k in range(p):
        s_tot = sumsq_tot[k] / t
        s1 = sumsq1[k] / n1
        s2 = sumsq2[k] / n2
        if s_tot <= 0 or s1 <= 0 or s2 <= 0:
            continue
        lr = t * np.log(s_tot) - n1 * np.log(s1) - n2 * np.log(s2)
        val = lr - 1.0
        if val > expected:
            expected = val

    assert np.isclose(result[0], expected)


def test_variance_no_alarm_univariate():
    """Single-feature Variance does not alarm on iid null stream."""
    rng = np.random.default_rng(99)
    x = rng.normal(loc=0.0, scale=1.0, size=200)
    detector = GridDetector(score=Variance(n_features=1), threshold=10.0)
    state = detector.init_state()
    for xi in x:
        state, out = detector.update(state, xi)
        assert not out["alarm"]


# ---------------------------------------------------------------------------
# MeanOrVariance parity tests
# ---------------------------------------------------------------------------


def test_mean_or_variance_alarm():
    """No alarm under null, alarm after a large mean shift."""
    n = 100
    p = 2
    rng = np.random.default_rng(11)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(score=MeanOrVariance(n_features=p), threshold=20.0)
    state, outputs = _run_stream(detector, x)
    assert not any(out["alarm"] for out in outputs)

    for _ in range(5):
        state, out = detector.update(state, np.array([50.0, 50.0]))
    assert out["alarm"]


def test_mean_or_variance_scores_match_formula():
    """_compute_centered_scores matches the per-feature mean+variance LR formula."""
    n = 60
    p = 2
    rng = np.random.default_rng(17)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    score = MeanOrVariance(n_features=p)
    total = score.init_state()
    for xi in x:
        total = score.update(total, xi)

    n1 = 20
    grid_state = score.init_state()
    for xi in x[:n1]:
        grid_state = score.update(grid_state, xi)

    result = score._compute_centered_scores(total, [grid_state])

    t = n
    n2 = t - n1
    s_tot = np.sum(x, axis=0)
    s1 = np.sum(x[:n1], axis=0)
    s2 = s_tot - s1
    ss_tot = np.sum(x**2, axis=0)
    ss1 = np.sum(x[:n1] ** 2, axis=0)
    ss2 = ss_tot - ss1

    expected = -1.0e300
    for k in range(p):
        sigma_tot = (ss_tot[k] - s_tot[k] ** 2 / t) / t
        sigma_1 = (ss1[k] - s1[k] ** 2 / n1) / n1
        sigma_2 = (ss2[k] - s2[k] ** 2 / n2) / n2
        if sigma_tot <= 0 or sigma_1 <= 0 or sigma_2 <= 0:
            continue
        lr = t * np.log(sigma_tot) - n1 * np.log(sigma_1) - n2 * np.log(sigma_2)
        val = lr - 2.0
        if val > expected:
            expected = val

    assert np.isclose(result[0], expected)


# ---------------------------------------------------------------------------
# RegressionMcScan parity tests
# ---------------------------------------------------------------------------


def test_regression_mcscan_alarm():
    """No alarm under null, alarm after a regression coefficient shift."""
    q = 3
    n = 100
    rng = np.random.default_rng(55)
    x_reg = rng.normal(size=(n, q))
    beta = np.zeros(q)
    y = x_reg @ beta + rng.normal(size=n)
    x = np.column_stack([y, x_reg])

    detector = GridDetector(score=RegressionMcScan(n_regressors=q), threshold=5.0)
    state, outputs = _run_stream(detector, x)
    assert not any(out["alarm"] for out in outputs)

    # Large coefficient shift: first regressor coefficient becomes 100.
    for _ in range(10):
        xi = rng.normal(size=q)
        yi = float(100.0 * xi[0] + rng.normal())
        state, out = detector.update(state, np.concatenate([[yi], xi]))
        if out["alarm"]:
            break
    assert out["alarm"]


def test_regression_mcscan_scores_match_formula():
    """_compute_centered_scores matches the McScan distance formula."""
    q = 3
    n = 60
    rng = np.random.default_rng(66)
    x_reg = rng.normal(size=(n, q))
    y = x_reg @ np.ones(q) + rng.normal(size=n)
    x = np.column_stack([y, x_reg])

    score = RegressionMcScan(n_regressors=q)
    total = score.init_state()
    for xi in x:
        total = score.update(total, xi)

    n1 = 25
    grid_state = score.init_state()
    for xi in x[:n1]:
        grid_state = score.update(grid_state, xi)

    result = score._compute_centered_scores(total, [grid_state])

    t = n
    n2 = t - n1
    yx_sum_pre = np.sum(y[:n1, None] * x_reg[:n1], axis=0)
    yx_sum_post = np.sum(y[n1:, None] * x_reg[n1:], axis=0)

    cov1 = yx_sum_pre / n1
    cov2 = yx_sum_post / n2
    expected = np.sqrt(n1 * n2 / t) * np.max(np.abs(cov1 - cov2))

    assert np.isclose(result[0], expected)


# ---------------------------------------------------------------------------
# RegressionDirect parity tests
# ---------------------------------------------------------------------------


def test_regression_direct_alarm():
    """No alarm under null, alarm after a large regression coefficient change."""
    q = 3
    n = 100
    rng = np.random.default_rng(77)
    x_reg = rng.normal(size=(n, q))
    y = x_reg @ np.zeros(q) + rng.normal(size=n)
    x = np.column_stack([y, x_reg])

    detector = GridDetector(score=RegressionDirect(n_regressors=q), threshold=10.0)
    state, outputs = _run_stream(detector, x)
    assert not any(out["alarm"] for out in outputs)

    for _ in range(20):
        xi = rng.normal(size=q)
        yi = float(1000.0 * xi[0] + rng.normal())
        state, out = detector.update(state, np.concatenate([[yi], xi]))
        if out["alarm"]:
            break
    assert out["alarm"]


def test_regression_direct_scores_match_formula():
    """_compute_centered_scores matches the inv_sqrtm_pd-based formula."""
    from gridcp.scores._score_helpers import inv_sqrtm_pd

    q = 3
    n = 60
    rng = np.random.default_rng(88)
    x_reg = rng.normal(size=(n, q))
    y = x_reg @ np.ones(q) + rng.normal(size=n)
    x = np.column_stack([y, x_reg])

    score = RegressionDirect(n_regressors=q)
    total = score.init_state()
    for xi in x:
        total = score.update(total, xi)

    n1 = 30
    grid_state = score.init_state()
    for xi in x[:n1]:
        grid_state = score.update(grid_state, xi)

    result = score._compute_centered_scores(total, [grid_state])

    yx_pre = np.sum(y[:n1, None] * x_reg[:n1], axis=0)
    yx_post = np.sum(y[n1:, None] * x_reg[n1:], axis=0)
    xx_pre = x_reg[:n1].T @ x_reg[:n1]
    xx_post = x_reg[n1:].T @ x_reg[n1:]

    m1_inv = inv_sqrtm_pd(xx_pre)
    m2_inv = inv_sqrtm_pd(xx_post)
    diff = m1_inv @ yx_pre - m2_inv @ yx_post
    expected = 0.5 * float(np.dot(diff, diff)) - q

    assert np.isclose(result[0], expected)


# ---------------------------------------------------------------------------
# MultivariateMeanIdentityCov parity test
# ---------------------------------------------------------------------------


def test_multivariate_mean_identity_cov_scores_match_formula():
    """_compute_centered_scores matches the sq-CUSUM formula."""
    p = 5
    n = 60
    rng = np.random.default_rng(33)
    x = rng.normal(size=(n, p))

    score = MultivariateMeanIdentityCov(n_features=p)
    total = score.init_state()
    for xi in x:
        total = score.update(total, xi)

    n1 = 20
    grid_state = score.init_state()
    for xi in x[:n1]:
        grid_state = score.update(grid_state, xi)

    result = score._compute_centered_scores(total, [grid_state])

    t = n
    n2 = t - n1
    s1 = np.sum(x[:n1], axis=0)
    s2 = np.sum(x[n1:], axis=0)
    bw = np.sqrt(n2 / (t * n1))
    aw = np.sqrt(n1 / (t * n2))
    sq_cusum = (bw * s1 - aw * s2) ** 2

    assert np.isclose(result[0, 0], float(np.max(sq_cusum)) - 1.0)
    assert np.isclose(result[0, 1], float(np.sum(sq_cusum)) - p)


# ---------------------------------------------------------------------------
# MultivariateMeanUnknownCov formula-parity test
# ---------------------------------------------------------------------------


def test_multivariate_mean_unknown_cov_scores_match_formula():
    """_compute_centered_scores matches the pooled-within-group LR formula."""
    p = 4
    # t must be >= 2*p+2 = 10 for the guard to pass.
    n = 60
    n1 = 20
    rng = np.random.default_rng(101)
    x = rng.normal(size=(n, p))

    score = MultivariateMeanUnknownCov(n_features=p)
    total = score.init_state()
    for xi in x:
        total = score.update(total, xi)

    grid_state = score.init_state()
    for xi in x[:n1]:
        grid_state = score.update(grid_state, xi)

    result = score._compute_centered_scores(total, [grid_state])

    # Reference formula (identical logic to original Python code).
    t = n
    n2 = t - n1
    s_tot = np.sum(x, axis=0)
    sxx_tot = x.T @ x

    sigma_null = (sxx_tot - np.outer(s_tot, s_tot) / t) / t
    _, logdet_null = np.linalg.slogdet(sigma_null)

    s1 = np.sum(x[:n1], axis=0)
    sxx1 = x[:n1].T @ x[:n1]
    s2 = s_tot - s1
    sxx2 = sxx_tot - sxx1

    sigma_alt = ((sxx1 - np.outer(s1, s1) / n1) + (sxx2 - np.outer(s2, s2) / n2)) / t
    _, logdet_alt = np.linalg.slogdet(sigma_alt)

    expected = t * (logdet_null - logdet_alt) - float(p)
    assert np.isclose(result[0], expected)


# ---------------------------------------------------------------------------
# MultivariateMeanOrCovariance formula-parity test
# ---------------------------------------------------------------------------


def test_multivariate_mean_or_covariance_scores_match_formula():
    """_compute_centered_scores matches the per-segment GLR formula."""
    p = 3
    # n1 and n2 must each be > 2*p = 6 for the guard to pass.
    n = 60
    n1 = 25
    rng = np.random.default_rng(202)
    x = rng.normal(size=(n, p))

    score = MultivariateMeanOrCovariance(n_features=p)
    total = score.init_state()
    for xi in x:
        total = score.update(total, xi)

    grid_state = score.init_state()
    for xi in x[:n1]:
        grid_state = score.update(grid_state, xi)

    result = score._compute_centered_scores(total, [grid_state])

    # Reference formula (identical logic to original Python code).
    t = n
    n2 = t - n1
    s_tot = np.sum(x, axis=0)
    sxx_tot = x.T @ x

    sigma_tot = (sxx_tot - np.outer(s_tot, s_tot) / t) / t
    _, logdet_tot = np.linalg.slogdet(sigma_tot)

    s1 = np.sum(x[:n1], axis=0)
    sxx1 = x[:n1].T @ x[:n1]
    s2 = s_tot - s1
    sxx2 = sxx_tot - sxx1

    sigma_1 = (sxx1 - np.outer(s1, s1) / n1) / n1
    sigma_2 = (sxx2 - np.outer(s2, s2) / n2) / n2
    _, logdet1 = np.linalg.slogdet(sigma_1)
    _, logdet2 = np.linalg.slogdet(sigma_2)

    df = float(p + (p * (p + 1)) // 2)
    expected = t * logdet_tot - n1 * logdet1 - n2 * logdet2 - df
    assert np.isclose(result[0], expected)
