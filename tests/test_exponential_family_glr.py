import numpy as np
import numba as nb
import pytest

from gridcp.detector import GridDetector
from gridcp.scores import ExponentialFamilyGLR
from gridcp.scores._exponential_family_glr import (
    make_newton_solver,
    make_vector_newton_solver,
)
from gridcp.scores._families import FAMILIES


# ---------------------------------------------------------------------------
# Reusable Numba callables — scalar Gaussian mean (v=1)
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
# Reusable Numba callables — MV Gaussian mean (v=p)
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
# Shared helper
# ---------------------------------------------------------------------------


def _detect(score, data, threshold):
    """Return True if the detector alarms at any point during `data`."""
    detector = GridDetector(score=score, threshold=threshold)
    state = detector.init_state()
    for x in data:
        obs = np.asarray([x]) if np.ndim(x) == 0 else np.asarray(x)
        state, out = detector.update(state, obs)
        if out["alarm"]:
            return True
    return False


# ---------------------------------------------------------------------------
# 1. Construction validation
# ---------------------------------------------------------------------------


def test_min_seg_below_2_raises():
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
    with pytest.raises(ValueError, match="A_prime and A_dprime are required"):
        ExponentialFamilyGLR(v=1, n_features=1, h=_h_gauss, A=_A_gauss)


def test_missing_A_grad_for_vector_raises():
    with pytest.raises(ValueError, match="A_grad and A_hess are required"):
        ExponentialFamilyGLR(v=3, n_features=3, h=_h_mv, A=_A_mv)


# ---------------------------------------------------------------------------
# 2. State lifecycle
# ---------------------------------------------------------------------------


def test_init_state_shape():
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


def test_update_accumulates_sufficient_statistic():
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


# ---------------------------------------------------------------------------
# 3. Scalar end-to-end — manual construction (v=1)
# ---------------------------------------------------------------------------


def test_scalar_detects_clear_mean_shift():
    rng = np.random.default_rng(seed=42)
    data = np.concatenate([rng.normal(0.0, 1.0, 50), rng.normal(10.0, 1.0, 50)])
    score = ExponentialFamilyGLR(
        v=1,
        n_features=1,
        h=_h_gauss,
        A=_A_gauss,
        A_prime=_Ap_gauss,
        A_dprime=_App_gauss,
    )
    assert _detect(score, data, threshold=5.0)


# ---------------------------------------------------------------------------
# 4. Vector end-to-end — manual construction (v>1)
# ---------------------------------------------------------------------------


def test_vector_detects_clear_mean_shift():
    rng = np.random.default_rng(seed=42)
    data = np.vstack(
        [rng.standard_normal((50, 2)), rng.standard_normal((50, 2)) + 10.0]
    )
    score = ExponentialFamilyGLR(
        v=2,
        n_features=2,
        h=_h_mv,
        A=_A_mv,
        A_grad=_Agrad_mv,
        A_hess=_Ahess_mv,
    )
    assert _detect(score, data, threshold=5.0)


# ---------------------------------------------------------------------------
# 5. Newton solver MLE correctness
# ---------------------------------------------------------------------------


def test_scalar_newton_finds_gaussian_mle():
    """Scalar solver recovers θ̂ = S/n = 2.5 for Gaussian mean family."""
    solver = make_newton_solver(_Ap_gauss, _App_gauss)
    theta = solver(25.0, 10.0, theta_init=0.0)
    assert np.isclose(theta, 2.5, atol=1e-7)


def test_vector_newton_finds_mv_gaussian_mle():
    """Vector solver recovers θ̂ = S/n for MV Gaussian mean (identity cov)."""
    solver = make_vector_newton_solver(_Agrad_mv, _Ahess_mv)
    theta = solver(np.array([10.0, 20.0]), 5.0, np.zeros(2))
    assert np.allclose(theta, [2.0, 4.0], atol=1e-7)


# ---------------------------------------------------------------------------
# 6. Domain constraint — backtracking stays within theta_max
# ---------------------------------------------------------------------------


def test_scalar_newton_stays_in_domain():
    """Solver with theta_max=0 returns a negative theta for the exponential family."""
    spec = FAMILIES["exponential"]
    solver = make_newton_solver(
        spec["A_prime"], spec["A_dprime"], theta_max=spec["theta_max"]
    )
    # A'(θ) = -1/θ = S/n  =>  θ_MLE = -n/S = -0.2
    theta = solver(50.0, 10.0, theta_init=-1.0)
    assert theta < 0.0
    assert np.isclose(theta, -0.2, atol=1e-7)


# ---------------------------------------------------------------------------
# 7. from_family() — error handling
# ---------------------------------------------------------------------------


def test_from_family_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown family"):
        ExponentialFamilyGLR.from_family("not_a_family")


def test_from_family_kwargs_forwarded():
    """gamma_rate accepts a shape kwarg via from_family."""
    score = ExponentialFamilyGLR.from_family("gamma_rate", shape=2.0)
    assert score.v == 1


# ---------------------------------------------------------------------------
# 8. from_family() — one minimal detection test per built-in family
# ---------------------------------------------------------------------------


def test_from_family_gaussian_mean_detects():
    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal(0.0, 1.0, 50), rng.normal(10.0, 1.0, 50)])
    score = ExponentialFamilyGLR.from_family("gaussian_mean")
    assert _detect(score, data, threshold=5.0)


def test_from_family_gaussian_variance_detects():
    rng = np.random.default_rng(1)
    # h(x) = x²; shift variance from 1 to 100
    data = np.concatenate([rng.normal(0.0, 1.0, 50), rng.normal(0.0, 10.0, 50)])
    score = ExponentialFamilyGLR.from_family("gaussian_variance")
    assert _detect(score, data, threshold=5.0)


def test_from_family_gaussian_mean_variance_detects():
    rng = np.random.default_rng(2)
    data = np.vstack([rng.normal(0.0, 1.0, (50, 1)), rng.normal(5.0, 5.0, (50, 1))])
    score = ExponentialFamilyGLR.from_family("gaussian_mean_variance")
    assert _detect(score, data, threshold=2.0)


def test_from_family_gaussian_covariance_detects():
    rng = np.random.default_rng(3)
    p = 2
    pre = rng.multivariate_normal(np.zeros(p), np.eye(p), 50)
    post = rng.multivariate_normal(np.zeros(p), np.diag([1.0, 100.0]), 50)
    data = np.vstack([pre, post])
    score = ExponentialFamilyGLR.from_family("gaussian_covariance", n_features=p)
    assert _detect(score, data, threshold=2.0)


def test_from_family_poisson_detects():
    rng = np.random.default_rng(4)
    data = np.concatenate(
        [rng.poisson(1.0, 50).astype(float), rng.poisson(20.0, 50).astype(float)]
    )
    score = ExponentialFamilyGLR.from_family("poisson")
    assert _detect(score, data, threshold=5.0)


def test_from_family_exponential_detects():
    rng = np.random.default_rng(5)
    # Shift rate from 1 (mean=1) to 10 (mean=0.1)
    data = np.concatenate([rng.exponential(1.0, 50), rng.exponential(0.1, 50)])
    score = ExponentialFamilyGLR.from_family("exponential")
    assert _detect(score, data, threshold=5.0)


def test_from_family_bernoulli_detects():
    rng = np.random.default_rng(6)
    data = np.concatenate(
        [rng.binomial(1, 0.1, 50).astype(float), rng.binomial(1, 0.9, 50).astype(float)]
    )
    score = ExponentialFamilyGLR.from_family("bernoulli")
    assert _detect(score, data, threshold=5.0)


def test_from_family_gamma_rate_detects():
    rng = np.random.default_rng(7)
    # Shift rate from 1 (scale=1) to 10 (scale=0.1), shape=3
    pre = rng.gamma(shape=3.0, scale=1.0, size=50)
    post = rng.gamma(shape=3.0, scale=0.1, size=50)
    data = np.concatenate([pre, post])
    score = ExponentialFamilyGLR.from_family("gamma_rate", shape=3.0)
    assert _detect(score, data, threshold=5.0)


# ---------------------------------------------------------------------------
# 9. min_seg regression tests
# ---------------------------------------------------------------------------


def test_gaussian_mean_mv_min_seg_is_2():
    """High-dimensional gaussian_mean must use min_seg=2 (not p+1).

    If min_seg were p+1, all candidates would be skipped on short streams
    and max_score would stay at 0.
    """
    p = 100
    score = ExponentialFamilyGLR.from_family("gaussian_mean", n_features=p)
    rng = np.random.default_rng(seed=0)
    data = np.vstack([rng.standard_normal((50, p)), rng.standard_normal((50, p)) + 3.0])
    detector = GridDetector(score=score, threshold=1e9)
    state = detector.init_state()
    max_scores = []
    for x in data:
        state, out = detector.update(state, x)
        max_scores.append(out["max_score"])
    assert max(max_scores) > 0.0, "All scores were 0 — min_seg is likely too large"


def test_gaussian_covariance_min_seg_is_p_plus_1():
    """gaussian_covariance must use min_seg=p+1 (not v+1 = p*(p+1)/2 + 1).

    If min_seg were v+1, all candidates would be skipped on short streams.
    """
    p = 3
    score = ExponentialFamilyGLR.from_family("gaussian_covariance", n_features=p)
    rng = np.random.default_rng(seed=1)
    n = 4 * (p + 1)
    pre = rng.multivariate_normal(np.zeros(p), np.eye(p), n // 2)
    post = rng.multivariate_normal(np.zeros(p), np.diag([1.0, 4.0, 9.0]), n // 2)
    data = np.vstack([pre, post])
    detector = GridDetector(score=score, threshold=1e9)
    state = detector.init_state()
    max_scores = []
    for x in data:
        state, out = detector.update(state, x)
        max_scores.append(out["max_score"])
    assert max(max_scores) > 0.0, "All scores were 0 — min_seg is likely too large"
