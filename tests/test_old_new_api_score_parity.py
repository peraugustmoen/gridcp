import numpy as np

import gridcp.old_api as old_api
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
):
    state = new_detector.init_state()
    running_new_max = 0.0

    for obs in X:
        old_detector.update(obs)
        state, out = new_detector.update(state, obs)
        ms = np.asarray(out["max_score"])
        if score_index is not None and ms.ndim > 0:
            ms = ms[score_index]
        running_new_max = max(running_new_max, float(ms))

        old_state = old_detector._state

        # Compare running sufficient statistics.
        old_running = _as_array(old_state["sum"])
        new_running = _as_array(running_from_new_state(state.running_score_state))
        assert np.allclose(old_running, new_running, atol=atol, rtol=rtol)

        # Compare candidate locations and candidate sufficient statistics.
        old_changelocs = _old_changelocs_from_grid(old_state["grid_list"])
        new_changelocs = np.asarray(state.grid, dtype=np.int64)
        assert np.array_equal(old_changelocs, new_changelocs)

        old_candidates = old_state["sum_pre_list"]
        new_candidates = state.candidate_score_states
        assert len(old_candidates) == len(new_candidates)

        for old_cand, new_cand in zip(old_candidates, new_candidates):
            old_arr = _as_array(old_cand)
            new_arr = _as_array(candidate_from_new_state(new_cand))
            assert np.allclose(old_arr, new_arr, atol=atol, rtol=rtol)

        # old_api maxx is running max over time; new_api max_score is per-time-step max.
        assert np.isclose(
            float(old_detector.max_statistic),
            running_new_max,
            atol=atol,
            rtol=rtol,
        )


def test_variance_parity_with_old_api():
    n = 40
    rng = np.random.default_rng(2026)
    x = rng.normal(0.0, 1.0, size=n)

    old_det = old_api.make_univariate_variance_change_detector(penalty_constant=1.0)
    new_det = GridDetector(score=Variance(n_features=1), threshold=1.0)

    _run_parity_check(
        X=x,
        old_detector=old_det,
        new_detector=new_det,
        running_from_new_state=lambda st: st.sum_sq,
        candidate_from_new_state=lambda st: st.sum_sq,
    )


def test_mean_or_variance_parity_with_old_api():
    n = 40
    rng = np.random.default_rng(2027)
    x = rng.normal(0.0, 1.0, size=n)

    old_det = old_api.make_univariate_mean_or_variance_change_detector(
        penalty_constant=1.0
    )
    new_det = GridDetector(score=MeanOrVariance(n_features=1), threshold=1.0)

    _run_parity_check(
        X=x,
        old_detector=old_det,
        new_detector=new_det,
        running_from_new_state=lambda st: np.array([st.sum[0], st.sum_sq[0]]),
        candidate_from_new_state=lambda st: np.array([st.sum[0], st.sum_sq[0]]),
    )


def test_multivariate_mean_identity_cov_parity_with_old_api():
    n = 40
    p = 4
    rng = np.random.default_rng(2028)
    x = rng.normal(0.0, 1.0, size=(n, p))

    old_det = old_api.make_multivariate_mean_change_detector(
        p=p,
        penalty_constant=1.0,
        mode="known_variance",
    )
    new_det = GridDetector(
        score=MultivariateMeanIdentityCov(n_features=p),
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
    n = 40
    p = 4
    rng = np.random.default_rng(2029)
    x = rng.normal(0.0, 1.0, size=(n, p))

    old_det = old_api.make_multivariate_mean_change_detector(
        p=p,
        penalty_constant=1.0,
        mode="unknown_variance",
    )
    new_det = GridDetector(
        score=MultivariateMeanUnknownCov(n_features=p), threshold=1.0
    )

    _run_parity_check(
        X=x,
        old_detector=old_det,
        new_detector=new_det,
        running_from_new_state=lambda st: np.vstack((st.sum, st.sum_outer)),
        candidate_from_new_state=lambda st: np.vstack((st.sum, st.sum_outer)),
    )


def test_multivariate_mean_or_covariance_parity_with_old_api():
    n = 40
    p = 4
    rng = np.random.default_rng(2030)
    x = rng.normal(0.0, 1.0, size=(n, p))

    old_det = old_api.make_multivariate_mean_or_covariance_change_detector(
        p=p,
        penalty_constant=1.0,
    )
    new_det = GridDetector(
        score=MultivariateMeanOrCovariance(n_features=p),
        threshold=1.0,
    )

    _run_parity_check(
        X=x,
        old_detector=old_det,
        new_detector=new_det,
        running_from_new_state=lambda st: np.vstack((st.sum, st.sum_outer)),
        candidate_from_new_state=lambda st: np.vstack((st.sum, st.sum_outer)),
    )


def test_regression_mcscan_parity_with_old_api():
    n = 40
    q = 4
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
    )


def test_regression_direct_parity_with_old_api():
    n = 40
    q = 4
    rng = np.random.default_rng(2032)
    x = rng.normal(0.0, 1.0, size=(n, q + 1))

    old_det = old_api.make_regression_change_detector(
        q=q,
        mode="direct",
        penalty_constant=1.0,
    )
    new_det = GridDetector(score=RegressionDirect(n_regressors=q), threshold=1.0)

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
    )
