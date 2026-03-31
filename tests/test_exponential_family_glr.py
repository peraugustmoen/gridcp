import numpy as np
import numba as nb
import pytest

from gridcp.detector import GridDetector
from gridcp.scores import ExponentialFamilyGLR
from gridcp.scores._exponential_family_glr import (
    make_newton_solver,
    make_vector_newton_solver,
)
from gridcp.scores._mean_cusum import MeanCUSUM


# ---------------------------------------------------------------------------
# Reusable exponential-family definitions for Gaussian mean (v=1)
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _h_gauss(x):
    return x[0]


@nb.njit(cache=True)
def _A_gauss(theta):
    return 0.5 * theta * theta


@nb.njit(cache=True)
def _Ap_gauss(theta):
    return theta


@nb.njit(cache=True)
def _App_gauss(theta):
    return 1.0


# ---------------------------------------------------------------------------
# Reusable exponential-family definitions for MV Gaussian mean (v=p)
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _h_mv(x):
    return x.copy()


@nb.njit(cache=True)
def _A_mv(theta):
    s = 0.0
    for i in range(theta.shape[0]):
        s += theta[i] * theta[i]
    return 0.5 * s


@nb.njit(cache=True)
def _Agrad_mv(theta):
    return theta.copy()


@nb.njit(cache=True)
def _Ahess_mv(theta):
    return np.eye(theta.shape[0])


# ---------------------------------------------------------------------------
# Shared helpers — match the _run_stream pattern from test_univariate_mean_change
# ---------------------------------------------------------------------------


def _run_stream_scalar(data, threshold=10.0):
    """Run data through a scalar (v=1) Gaussian-mean EF GLR detector."""
    score = ExponentialFamilyGLR(
        v=1,
        n_features=1,
        h=_h_gauss,
        A=_A_gauss,
        A_prime=_Ap_gauss,
        A_dprime=_App_gauss,
    )
    detector = GridDetector(score=score, threshold=threshold)
    state = detector.init_state()
    outputs = []
    for x in data:
        state, out = detector.update(state, np.asarray([x]))
        outputs.append(out)
    return detector, state, outputs


def _run_stream_mv(data, n_features=3, threshold=10.0):
    """Run data through a multivariate (v=p) Gaussian-mean EF GLR detector."""
    score = ExponentialFamilyGLR(
        v=n_features,
        n_features=n_features,
        h=_h_mv,
        A=_A_mv,
        A_grad=_Agrad_mv,
        A_hess=_Ahess_mv,
    )
    detector = GridDetector(score=score, threshold=threshold)
    state = detector.init_state()
    outputs = []
    for x in data:
        state, out = detector.update(state, x)
        outputs.append(out)
    return detector, state, outputs


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_min_seg_below_2_raises():
    """Reject min_seg values below 2."""
    with pytest.raises(ValueError, match="min_seg must be >= 2"):
        ExponentialFamilyGLR(
            v=1,
            n_features=1,
            h=_h_gauss,
            A=_A_gauss,
            A_prime=_Ap_gauss,
            A_dprime=_App_gauss,
            min_seg=1,
        )


def test_missing_A_prime_for_scalar_raises():
    """Require A_prime and A_dprime for the scalar case (v=1)."""
    with pytest.raises(ValueError, match="A_prime and A_dprime are required"):
        ExponentialFamilyGLR(
            v=1,
            n_features=1,
            h=_h_gauss,
            A=_A_gauss,
        )


def test_missing_A_grad_for_vector_raises():
    """Require A_grad and A_hess for the vector case (v>1)."""
    with pytest.raises(ValueError, match="A_grad and A_hess are required"):
        ExponentialFamilyGLR(
            v=3,
            n_features=3,
            h=_h_mv,
            A=_A_mv,
        )


# ---------------------------------------------------------------------------
# State lifecycle
# ---------------------------------------------------------------------------


def test_init_state_scalar_shape():
    """Scalar init_state returns suff_stat of shape (1,) with zero."""
    score = ExponentialFamilyGLR(
        v=1,
        n_features=1,
        h=_h_gauss,
        A=_A_gauss,
        A_prime=_Ap_gauss,
        A_dprime=_App_gauss,
    )
    state = score.init_state()
    assert state.n_samples == 0
    assert state.suff_stat.shape == (1,)
    assert np.allclose(state.suff_stat, 0.0)


def test_init_state_vector_shape():
    """Vector init_state returns suff_stat of correct dimension."""
    score = ExponentialFamilyGLR(
        v=3,
        n_features=3,
        h=_h_mv,
        A=_A_mv,
        A_grad=_Agrad_mv,
        A_hess=_Ahess_mv,
    )
    state = score.init_state()
    assert state.suff_stat.shape == (3,)


def test_update_accumulates_sufficient_statistic():
    """Successive updates accumulate h(x) in the sufficient statistic."""
    score = ExponentialFamilyGLR(
        v=1,
        n_features=1,
        h=_h_gauss,
        A=_A_gauss,
        A_prime=_Ap_gauss,
        A_dprime=_App_gauss,
    )
    state = score.init_state()
    state = score.update(state, np.array([2.5]))
    assert state.n_samples == 1
    assert np.isclose(state.suff_stat[0], 2.5)

    state = score.update(state, np.array([1.5]))
    assert state.n_samples == 2
    assert np.isclose(state.suff_stat[0], 4.0)


def test_update_wrong_observation_size_raises():
    """Reject observations whose size does not match n_features."""
    score = ExponentialFamilyGLR(
        v=1,
        n_features=1,
        h=_h_gauss,
        A=_A_gauss,
        A_prime=_Ap_gauss,
        A_dprime=_App_gauss,
    )
    state = score.init_state()
    with pytest.raises(ValueError, match="Expected observation of size"):
        score.update(state, np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Scalar (v=1) Gaussian mean — end-to-end
# ---------------------------------------------------------------------------


def test_scalar_no_false_alarm_under_null():
    """Avoid false alarms under null data with a high threshold."""
    rng = np.random.default_rng(seed=42)
    data = rng.normal(0.0, 1.0, size=200)

    _, _, outputs = _run_stream_scalar(data, threshold=50.0)
    assert all(not out["alarm"] for out in outputs)


def test_scalar_detects_clear_mean_shift():
    """Detect a clear mean shift introduced mid-stream."""
    rng = np.random.default_rng(seed=42)
    pre = rng.normal(0.0, 1.0, size=100)
    post = rng.normal(5.0, 1.0, size=100)
    data = np.concatenate([pre, post])

    _, _, outputs = _run_stream_scalar(data, threshold=5.0)
    post_shift_outputs = outputs[len(pre) :]
    assert any(out["alarm"] for out in post_shift_outputs)


# ---------------------------------------------------------------------------
# Vector (v>1) MV Gaussian mean — end-to-end
# ---------------------------------------------------------------------------


def test_mv_no_false_alarm_under_null():
    """Avoid false alarms for multivariate null data with a high threshold."""
    rng = np.random.default_rng(seed=42)
    data = rng.standard_normal((200, 3))

    _, _, outputs = _run_stream_mv(data, n_features=3, threshold=50.0)
    assert all(not out["alarm"] for out in outputs)


def test_mv_detects_clear_mean_shift():
    """Detect a clear multivariate mean shift mid-stream."""
    rng = np.random.default_rng(seed=42)
    pre = rng.standard_normal((100, 3))
    post = rng.standard_normal((200, 3)) + 3.0
    data = np.concatenate([pre, post])

    _, _, outputs = _run_stream_mv(data, n_features=3, threshold=5.0)
    post_shift_outputs = outputs[len(pre) :]
    assert any(out["alarm"] for out in post_shift_outputs)


# ---------------------------------------------------------------------------
# Equivalence with MeanCUSUM for univariate Gaussian mean
# ---------------------------------------------------------------------------
def test_ef_and_mean_cusum_both_detect_same_shift():
    """Both EF GLR and MeanCUSUM should detect the same univariate mean shift."""
    rng = np.random.default_rng(seed=99)
    pre = rng.normal(0.0, 1.0, size=100)
    post = rng.normal(3.0, 1.0, size=100)
    data = np.concatenate([pre, post])

    score_ef = ExponentialFamilyGLR(
        v=1,
        n_features=1,
        h=_h_gauss,
        A=_A_gauss,
        A_prime=_Ap_gauss,
        A_dprime=_App_gauss,
    )
    score_ref = MeanCUSUM(n_features=1)

    det_ef = GridDetector(score=score_ef, threshold=5.0)
    det_ref = GridDetector(score=score_ref, threshold=5.0)

    state_ef = det_ef.init_state()
    state_ref = det_ref.init_state()
    alarm_ef = alarm_ref = None

    for i, x in enumerate(data):
        obs = np.array([x])
        state_ef, out_ef = det_ef.update(state_ef, obs)
        state_ref, out_ref = det_ref.update(state_ref, obs)
        if alarm_ef is None and out_ef["alarm"]:
            alarm_ef = i
        if alarm_ref is None and out_ref["alarm"]:
            alarm_ref = i

    assert alarm_ef is not None, "EF should have alarmed"
    assert alarm_ref is not None, "MeanCUSUM should have alarmed"
    # Both should detect within a reasonable window of each other
    assert abs(alarm_ef - alarm_ref) < 30


# ---------------------------------------------------------------------------
# Scalar Newton solver (make_newton_solver)
# ---------------------------------------------------------------------------


def test_scalar_newton_finds_gaussian_mle():
    """Scalar solver recovers MLE θ̂ = S/n = x̄ for Gaussian mean family."""
    solver = make_newton_solver(_Ap_gauss, _App_gauss)
    # S = sum of observations, n = count  =>  MLE = S / n
    S, n = 25.0, 10.0
    theta = solver(S, n, theta_init=0.0)
    assert np.isclose(theta, 2.5, atol=1e-7)


def test_scalar_newton_negative_mean():
    """Scalar solver finds negative MLE correctly."""
    solver = make_newton_solver(_Ap_gauss, _App_gauss)
    S, n = -30.0, 10.0
    theta = solver(S, n, theta_init=0.0)
    assert np.isclose(theta, -3.0, atol=1e-7)


def test_scalar_newton_converges_from_distant_init():
    """Scalar solver converges even when theta_init is far from MLE."""
    solver = make_newton_solver(_Ap_gauss, _App_gauss)
    S, n = 5.0, 5.0  # MLE = 1.0
    theta = solver(S, n, theta_init=100.0)
    assert np.isclose(theta, 1.0, atol=1e-7)


def test_scalar_newton_single_observation():
    """Scalar solver works for n=1 (single observation)."""
    solver = make_newton_solver(_Ap_gauss, _App_gauss)
    S, n = 7.3, 1.0
    theta = solver(S, n, theta_init=0.0)
    assert np.isclose(theta, 7.3, atol=1e-7)


def test_scalar_newton_domain_guard_negative_init():
    """Scalar solver with negative theta_init does not cross zero."""

    # Exponential family for variance: A(θ) = -0.5 * log(-θ), θ < 0
    # A'(θ) = -1 / (2θ),  A''(θ) = 1 / (2θ²)
    @nb.njit(cache=True)
    def Ap_var(theta):
        return -1.0 / (2.0 * theta)

    @nb.njit(cache=True)
    def App_var(theta):
        return 1.0 / (2.0 * theta * theta)

    solver = make_newton_solver(Ap_var, App_var)
    # MLE for A'(θ) = S/n  =>  -1/(2θ) = S/n  =>  θ = -n/(2S)
    # With S = 50, n = 10: θ_MLE = -10/100 = -0.1
    theta = solver(50.0, 10.0, theta_init=-1.0)
    assert theta < 0.0, "Domain guard should keep theta negative"
    assert np.isclose(theta, -0.1, atol=1e-6)


# ---------------------------------------------------------------------------
# Vector Newton solver (make_vector_newton_solver)
# ---------------------------------------------------------------------------


def test_vector_newton_finds_mv_gaussian_mle():
    """Vector solver recovers MLE θ̂ = S/n for MV Gaussian mean (identity cov)."""
    solver = make_vector_newton_solver(_Agrad_mv, _Ahess_mv)
    S_vec = np.array([10.0, 20.0, 30.0])
    n = 5.0
    theta_init = np.zeros(3)
    theta = solver(S_vec, n, theta_init)
    expected = S_vec / n  # [2, 4, 6]
    assert np.allclose(theta, expected, atol=1e-7)


def test_vector_newton_single_observation():
    """Vector solver works for n=1."""
    solver = make_vector_newton_solver(_Agrad_mv, _Ahess_mv)
    S_vec = np.array([3.14, -2.72])
    theta_init = np.zeros(2)
    theta = solver(S_vec, 1.0, theta_init)
    assert np.allclose(theta, S_vec, atol=1e-7)


def test_vector_newton_converges_from_distant_init():
    """Vector solver converges even when theta_init is far from MLE."""
    solver = make_vector_newton_solver(_Agrad_mv, _Ahess_mv)
    S_vec = np.array([5.0, 10.0, 15.0])
    n = 5.0
    theta_init = np.array([100.0, -100.0, 50.0])
    theta = solver(S_vec, n, theta_init)
    assert np.allclose(theta, np.array([1.0, 2.0, 3.0]), atol=1e-7)


def test_vector_newton_high_dimension():
    """Vector solver works for a larger dimension (p=10)."""
    solver = make_vector_newton_solver(_Agrad_mv, _Ahess_mv)
    rng = np.random.default_rng(seed=123)
    S_vec = rng.standard_normal(10) * 50
    n = 20.0
    theta_init = np.zeros(10)
    theta = solver(S_vec, n, theta_init)
    assert np.allclose(theta, S_vec / n, atol=1e-7)
