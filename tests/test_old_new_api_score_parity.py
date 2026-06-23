import numpy as np

import gridcp.old_api as old_api
from gridcp.detector import GridDetector
from gridcp.scores import (
    CUSUM,
    GaussianMean,
    GaussianMeanOrVariance,
    GaussianMeanOrCovariance,
    GaussianMeanFullCovariance,
    RegressionWald,
    RegressionMcScan,
    GaussianVariance,
)


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _old_changelocs_from_grid(old_grid) -> np.ndarray:
    # old_api stores shifted grid points: g_shift = G^(t) - t - 1.
    # changeloc = t - G^(t) = -g_shift - 1.
    return np.asarray([-int(g_shift) - 1 for g_shift in old_grid], dtype=np.int64)


def _run_parity_check(
    X,
    old_detector,
    new_detector,
    running_from_new_state,
    candidate_from_new_state,
    atol: float = 1e-8,
    rtol: float = 1e-8,
    score_index: int | None = None,
    check_max_statistic: bool = True,
):
    state = new_detector.init_state()
    running_new_max = 0.0

    for obs in X:
        old_detector.update(obs)
        state, out = new_detector.update(state, obs)
        ms = np.asarray(out["max_score"], dtype=np.float64)
        if score_index is not None:
            score_value = float(ms[score_index])
        else:
            score_value = float(ms[0])
        running_new_max = max(running_new_max, score_value)

        old_state = old_detector._state

        # Compare running sufficient statistics.
        old_running = _as_array(old_state["sum"])
        new_running = _as_array(running_from_new_state(state.current_score_state))
        assert np.allclose(old_running, new_running, atol=atol, rtol=rtol)

        # Compare candidate locations and candidate sufficient statistics.
        old_changelocs = _old_changelocs_from_grid(old_state["grid_list"])
        new_changelocs = np.asarray(state.grid, dtype=np.int64)
        assert np.array_equal(old_changelocs, new_changelocs)

        old_candidates = old_state["sum_pre_list"]
        new_candidates = state.previous_score_states
        assert len(old_candidates) == len(new_candidates)

        for old_cand, new_cand in zip(old_candidates, new_candidates):
            old_arr = _as_array(old_cand)
            new_arr = _as_array(candidate_from_new_state(new_cand))
            assert np.allclose(old_arr, new_arr, atol=atol, rtol=rtol)

        # old_api maxx is running max over time; new_api max_score is per-time-step max.
        if check_max_statistic:
            assert np.isclose(
                float(old_detector.max_statistic),
                running_new_max,
                atol=atol,
                rtol=rtol,
            )


def test_variance_parity_with_old_api():
    n = 24
    rng = np.random.default_rng(2026)
    x = rng.normal(0.0, 1.0, size=n)

    old_det = old_api.make_univariate_variance_change_detector(penalty_constant=1.0)
    new_det = GridDetector(score=GaussianVariance(n_features=1), threshold=1.0)

    _run_parity_check(
        X=x,
        old_detector=old_det,
        new_detector=new_det,
        running_from_new_state=lambda st: st.sum_sq,
        candidate_from_new_state=lambda st: st.sum_sq,
    )


def test_mean_or_variance_parity_with_old_api():
    n = 24
    rng = np.random.default_rng(2027)
    x = rng.normal(0.0, 1.0, size=n)

    old_det = old_api.make_univariate_mean_or_variance_change_detector(
        penalty_constant=1.0
    )
    new_det = GridDetector(score=GaussianMeanOrVariance(n_features=1), threshold=1.0)

    _run_parity_check(
        X=x,
        old_detector=old_det,
        new_detector=new_det,
        running_from_new_state=lambda st: np.array([st.sum[0], st.sum_sq[0]]),
        candidate_from_new_state=lambda st: np.array([st.sum[0], st.sum_sq[0]]),
        check_max_statistic=False,
    )


def test_multivariate_mean_identity_cov_parity_with_old_api():
    n = 24
    p = 3
    rng = np.random.default_rng(2028)
    x = rng.normal(0.0, 1.0, size=(n, p))

    old_det = old_api.make_multivariate_mean_change_detector(
        p=p,
        penalty_constant=1.0,
        mode="known_variance",
    )
    new_det = GridDetector(
        score=CUSUM(n_features=p, aggregation="max-sum"),
        threshold=np.array([1.0, 1.0], dtype=np.float64),
    )

    _run_parity_check(
        X=x,
        old_detector=old_det,
        new_detector=new_det,
        running_from_new_state=lambda st: st.sum,
        candidate_from_new_state=lambda st: st.sum,
        score_index=1,
    )


def test_multivariate_mean_unknown_cov_parity_with_old_api():
    n = 24
    p = 3
    rng = np.random.default_rng(2029)
    x = rng.normal(0.0, 1.0, size=(n, p))

    old_det = old_api.make_multivariate_mean_change_detector(
        p=p,
        penalty_constant=1.0,
        mode="unknown_variance",
    )
    new_det = GridDetector(
        score=GaussianMeanFullCovariance(n_features=p), threshold=1.0
    )

    _run_parity_check(
        X=x,
        old_detector=old_det,
        new_detector=new_det,
        running_from_new_state=lambda st: np.vstack((st.sum, st.sum_outer)),
        candidate_from_new_state=lambda st: np.vstack((st.sum, st.sum_outer)),
        check_max_statistic=False,
    )


def test_multivariate_mean_or_covariance_parity_with_old_api():
    n = 24
    p = 3
    rng = np.random.default_rng(2030)
    x = rng.normal(0.0, 1.0, size=(n, p))

    old_det = old_api.make_multivariate_mean_or_covariance_change_detector(
        p=p,
        penalty_constant=1.0,
    )
    new_det = GridDetector(
        score=GaussianMeanOrCovariance(n_features=p),
        threshold=1.0,
    )

    _run_parity_check(
        X=x,
        old_detector=old_det,
        new_detector=new_det,
        running_from_new_state=lambda st: np.vstack((st.sum, st.sum_outer)),
        candidate_from_new_state=lambda st: np.vstack((st.sum, st.sum_outer)),
        check_max_statistic=False,
    )


def test_regression_mcscan_parity_with_old_api():
    n = 24
    q = 3
    rng = np.random.default_rng(2031)
    x = rng.normal(0.0, 1.0, size=(n, q + 1))

    old_det = old_api.make_regression_change_detector(
        q=q,
        mode="McScan",
        penalty_constant=1.0,
    )
    new_det = GridDetector(score=RegressionMcScan(n_regressors=q), threshold=1.0)

    _run_parity_check(
        X=x,
        old_detector=old_det,
        new_detector=new_det,
        running_from_new_state=lambda st: st.yx_sum,
        candidate_from_new_state=lambda st: st.yx_sum,
        check_max_statistic=False,
    )


def test_regression_direct_parity_with_old_api():
    n = 24
    q = 3
    rng = np.random.default_rng(2032)
    x = rng.normal(0.0, 1.0, size=(n, q + 1))

    old_det = old_api.make_regression_change_detector(
        q=q,
        mode="direct",
        penalty_constant=1.0,
    )
    new_det = GridDetector(score=RegressionWald(n_regressors=q), threshold=1.0)

    # The new API uses guard n1 < q (Gram matrix full rank at n1 = q), while
    # the old API used n1 <= q (off by one, too conservative).  The running
    # sufficient statistics and grid positions still match; only the max
    # statistic can differ at exactly n1 = q.
    _run_parity_check(
        X=x,
        old_detector=old_det,
        new_detector=new_det,
        running_from_new_state=lambda st: np.concatenate(
            [st.yx_sum, st.xx_sum.reshape(-1)]
        ),
        candidate_from_new_state=lambda st: np.concatenate(
            [st.yx_sum, st.xx_sum.reshape(-1)]
        ),
        check_max_statistic=False,
    )


def test_mean_cusum_unknown_variance_parity_with_old_api():
    """GaussianMean running sums and grid candidates match old API."""
    n = 24
    rng = np.random.default_rng(2033)
    x = rng.normal(0.0, 1.0, size=n)

    old_det = old_api.make_univariate_mean_change_detector(
        penalty_constant=1.0, mode="unknown_variance"
    )
    new_det = GridDetector(score=GaussianMean(n_features=1), threshold=1.0)

    # Old API stores [sum(x), sum(x^2)] as shape (2,); new API stores the
    # same in state.stats with shape (2,) for univariate.
    _run_parity_check(
        X=x,
        old_detector=old_det,
        new_detector=new_det,
        running_from_new_state=lambda st: st.stats,
        candidate_from_new_state=lambda st: st.stats,
        check_max_statistic=False,
    )
