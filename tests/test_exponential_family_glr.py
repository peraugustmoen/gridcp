import numpy as np
import numba as nb
import pytest

from gridcp.detector import GridDetector
from gridcp.scores import ExponentialFamilyGLR, PenaltyType
from gridcp.scores._exponential_family_glr import (
    make_newton_solver,
    make_vector_newton_solver,
)
from gridcp.scores._families import FAMILIES


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
# Scalar Newton solver (make_newton_solver)
# ---------------------------------------------------------------------------


def test_scalar_newton_finds_gaussian_mle():
    """Scalar solver recovers MLE θ̂ = S/n = x̄ for Gaussian mean family."""
    solver = make_newton_solver(_Ap_gauss, _App_gauss)
    # S = sum of observations, n = count  =>  MLE = S / n
    S, n = 25.0, 10.0
    theta = solver(S, n, theta_init=0.0)
    assert np.isclose(theta, 2.5, atol=1e-7)


def test_scalar_newton_stays_in_domain_negative_init():
    """Scalar solver with negative theta_init stays negative via backtracking."""

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
    assert theta < 0.0, "Backtracking should keep theta in the valid domain (θ < 0)"
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


# ---------------------------------------------------------------------------
# from_family — construction and error handling
# ---------------------------------------------------------------------------


def test_from_family_unknown_name_raises():
    """Reject unrecognised family names with a helpful message."""
    with pytest.raises(ValueError, match="Unknown family"):
        ExponentialFamilyGLR.from_family("gamma")


def test_from_family_error_message_lists_valid_families():
    """Error message should mention at least one known family name."""
    with pytest.raises(ValueError, match="bernoulli"):
        ExponentialFamilyGLR.from_family("bad_name")


def test_from_family_theta_init_override():
    """Caller can override the family's default theta_init."""
    score = ExponentialFamilyGLR.from_family("exponential", theta_init=-2.0)
    assert np.isclose(score._theta_init, -2.0)


def test_from_family_penalty_forwarded():
    """penalty kwarg is forwarded to the constructed object."""
    score = ExponentialFamilyGLR.from_family("poisson", penalty=PenaltyType.CONSTANT)
    assert score.penalty == PenaltyType.CONSTANT


# ---------------------------------------------------------------------------
# from_family — smoke tests (all families construct without error)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family",
    ["bernoulli", "exponential", "gaussian_mean", "gaussian_variance", "poisson"],
)
def test_from_family_scalar_constructs(family):
    """All built-in scalar (v=1) families construct without error."""
    score = ExponentialFamilyGLR.from_family(family)
    assert score.v == 1


@pytest.mark.parametrize("p", [2, 3, 5])
def test_from_family_gaussian_mean_mv_constructs(p):
    """gaussian_mean with n_features=p constructs with v=p."""
    score = ExponentialFamilyGLR.from_family("gaussian_mean", n_features=p)
    assert score.v == p


@pytest.mark.parametrize("p", [2, 3, 5])
def test_from_family_gaussian_covariance_attributes(p):
    """gaussian_covariance with n_features=p constructs with v=p*(p+1)//2."""
    score = ExponentialFamilyGLR.from_family("gaussian_covariance", n_features=p)
    assert score.v == p * (p + 1) // 2
    assert score.n_features == p


# ---------------------------------------------------------------------------
# from_family — MLE correctness (Newton solver, closed-form checks)
# ---------------------------------------------------------------------------


def test_from_family_poisson_mle():
    """Poisson MLE: A'(theta)=exp(theta)=S/n => theta=log(S/n)."""
    spec = FAMILIES["poisson"]
    solver = make_newton_solver(spec["A_prime"], spec["A_dprime"])
    S, n = 30.0, 10.0
    theta = solver(S, n, theta_init=0.0)
    assert np.isclose(theta, np.log(3.0), atol=1e-7)


def test_from_family_bernoulli_mle():
    """Bernoulli MLE: sigmoid(theta)=S/n => theta=log(S/n / (1-S/n))."""
    spec = FAMILIES["bernoulli"]
    solver = make_newton_solver(spec["A_prime"], spec["A_dprime"])
    S, n = 7.0, 10.0  # p=0.7 => theta = log(7/3)
    theta = solver(S, n, theta_init=0.0)
    assert np.isclose(theta, np.log(7.0 / 3.0), atol=1e-6)


def test_from_family_exponential_mle():
    """Exponential MLE: A'(theta)=-1/theta=S/n => theta=-n/S."""
    spec = FAMILIES["exponential"]
    solver = make_newton_solver(
        spec["A_prime"], spec["A_dprime"], theta_max=spec["theta_max"]
    )
    S, n = 50.0, 10.0  # theta_MLE = -10/50 = -0.2
    theta = solver(S, n, theta_init=-1.0)
    assert np.isclose(theta, -0.2, atol=1e-7)
    assert theta < 0.0


def test_from_family_gaussian_variance_mle():
    """Gaussian variance MLE: A'(theta)=-1/(2*theta)=S/n => theta=-n/(2*S)."""
    spec = FAMILIES["gaussian_variance"]
    solver = make_newton_solver(
        spec["A_prime"], spec["A_dprime"], theta_max=spec["theta_max"]
    )
    S, n = 40.0, 10.0  # theta_MLE = -10/80 = -0.125
    theta = solver(S, n, theta_init=-0.5)
    assert np.isclose(theta, -0.125, atol=1e-7)
    assert theta < 0.0


def test_from_family_gamma_rate_mle():
    """Gamma-rate MLE: A'(theta)=-k/theta=S/n => theta=-k*n/S."""
    # With k=3, S=30, n=10: theta_MLE = -3*10/30 = -1.0
    spec = FAMILIES["gamma_rate"](1, shape=3.0)
    solver = make_newton_solver(
        spec["A_prime"], spec["A_dprime"], theta_max=spec["theta_max"]
    )
    S, n = 30.0, 10.0
    theta = solver(S, n, theta_init=-1.0)
    assert np.isclose(theta, -1.0, atol=1e-7)
    assert theta < 0.0


# ---------------------------------------------------------------------------
# from_family — end-to-end detection
# ---------------------------------------------------------------------------


def test_from_family_gaussian_mean_detects_shift():
    """gaussian_mean factory detects a clear mean shift."""
    rng = np.random.default_rng(seed=1)
    data = np.concatenate([rng.normal(0.0, 1.0, 100), rng.normal(5.0, 1.0, 100)])
    score = ExponentialFamilyGLR.from_family("gaussian_mean")
    detector = GridDetector(score=score, threshold=5.0)
    state = detector.init_state()
    post_shift_alarms = []
    for i, x in enumerate(data):
        state, out = detector.update(state, np.array([x]))
        if i >= 100 and out["alarm"]:
            post_shift_alarms.append(i)
    assert len(post_shift_alarms) > 0


def test_from_family_gaussian_mean_mv_detects_shift():
    """gaussian_mean (multivariate) factory detects a multivariate mean shift."""
    rng = np.random.default_rng(seed=2)
    data = np.vstack(
        [rng.standard_normal((100, 3)), rng.standard_normal((100, 3)) + 3.0]
    )
    score = ExponentialFamilyGLR.from_family("gaussian_mean", n_features=3)
    detector = GridDetector(score=score, threshold=5.0)
    state = detector.init_state()
    post_shift_alarms = []
    for i, x in enumerate(data):
        state, out = detector.update(state, x)
        if i >= 100 and out["alarm"]:
            post_shift_alarms.append(i)
    assert len(post_shift_alarms) > 0


def test_from_family_gamma_rate_detects_rate_shift():
    """gamma_rate factory detects a change in rate with known shape."""
    rng = np.random.default_rng(seed=42)
    # Pre: Gamma(shape=3, rate=1) → mean=3; Post: Gamma(shape=3, rate=3) → mean=1
    pre = rng.gamma(shape=3.0, scale=1.0, size=150).astype(float)  # scale=1/rate
    post = rng.gamma(shape=3.0, scale=1.0 / 3.0, size=150).astype(float)
    data = np.concatenate([pre, post])
    score = ExponentialFamilyGLR.from_family("gamma_rate", shape=3.0)
    detector = GridDetector(score=score, threshold=5.0)
    state = detector.init_state()
    post_shift_alarms = []
    for i, x in enumerate(data):
        state, out = detector.update(state, np.array([x]))
        if i >= 150 and out["alarm"]:
            post_shift_alarms.append(i)
    assert len(post_shift_alarms) > 0


def test_from_family_poisson_detects_rate_shift():
    """poisson factory detects a clear Poisson rate shift."""
    rng = np.random.default_rng(seed=3)
    data = np.concatenate(
        [rng.poisson(2.0, 150).astype(float), rng.poisson(8.0, 150).astype(float)]
    )
    score = ExponentialFamilyGLR.from_family("poisson")
    detector = GridDetector(score=score, threshold=5.0)
    state = detector.init_state()
    post_shift_alarms = []
    for i, x in enumerate(data):
        state, out = detector.update(state, np.array([x]))
        if i >= 150 and out["alarm"]:
            post_shift_alarms.append(i)
    assert len(post_shift_alarms) > 0


# ---------------------------------------------------------------------------
# from_family — parity with manual construction for gaussian_mean
# ---------------------------------------------------------------------------


def test_from_family_gaussian_mean_parity_with_manual():
    """from_family('gaussian_mean') and manual construction detect at same step."""
    rng = np.random.default_rng(seed=42)
    data = np.concatenate([rng.normal(0.0, 1.0, 100), rng.normal(4.0, 1.0, 100)])

    score_manual = ExponentialFamilyGLR(
        v=1,
        n_features=1,
        h=_h_gauss,
        A=_A_gauss,
        A_prime=_Ap_gauss,
        A_dprime=_App_gauss,
    )
    score_factory = ExponentialFamilyGLR.from_family("gaussian_mean")

    def first_alarm(score):
        detector = GridDetector(score=score, threshold=5.0)
        state = detector.init_state()
        for i, x in enumerate(data):
            state, out = detector.update(state, np.array([x]))
            if out["alarm"]:
                return i
        return None

    alarm_manual = first_alarm(score_manual)
    alarm_factory = first_alarm(score_factory)
    assert alarm_manual is not None and alarm_factory is not None
    assert alarm_manual == alarm_factory


# ---------------------------------------------------------------------------
# gaussian_mean_variance — MLE and end-to-end
# ---------------------------------------------------------------------------


def test_gaussian_mean_variance_mle():
    """MLE for Gaussian mean+variance: ∇A(θ̂) = S/n has closed form.

    For N(mu, sigma^2): theta0 = mu/sigma^2, theta1 = -1/(2*sigma^2).
    Given sufficient stats S0 = sum(x), S1 = sum(x^2) with n observations,
    the MLE satisfies mu_hat = S0/n, sigma2_hat = S1/n - (S0/n)^2.
    """
    from gridcp.scores._exponential_family_glr import make_vector_newton_solver

    spec = FAMILIES["gaussian_mean_variance"]
    solver = make_vector_newton_solver(spec["A_grad"], spec["A_hess"])

    # Generate data from N(2, 4) -> theta = (2/4, -1/8) = (0.5, -0.125)
    rng = np.random.default_rng(0)
    data = rng.normal(2.0, 2.0, 1000)
    S = np.array([data.sum(), (data**2).sum()])
    n = float(len(data))

    theta_init = np.array([0.0, -0.5])
    theta = solver(S, n, theta_init)

    # theta0 = mu/sigma^2 = 2/4 = 0.5, theta1 = -1/(2*4) = -0.125
    assert np.allclose(theta, [0.5, -0.125], atol=0.05)
    assert theta[1] < 0.0


def test_from_family_gaussian_mean_variance_detects_joint_shift():
    """gaussian_mean_variance detects a joint change in mean and variance."""
    rng = np.random.default_rng(seed=7)
    pre = rng.normal(0.0, 1.0, (150, 1))
    post = rng.normal(2.0, 2.0, (150, 1))
    data = np.vstack([pre, post])
    score = ExponentialFamilyGLR.from_family("gaussian_mean_variance")
    detector = GridDetector(score=score, threshold=2.0)
    state = detector.init_state()
    post_shift_alarms = []
    for i, x in enumerate(data):
        state, out = detector.update(state, x)
        if i >= 150 and out["alarm"]:
            post_shift_alarms.append(i)
    assert len(post_shift_alarms) > 0


def test_from_family_gaussian_mean_variance_no_false_alarm():
    """gaussian_mean_variance does not alarm under null with high threshold."""
    rng = np.random.default_rng(seed=8)
    data = rng.normal(0.0, 1.0, (300, 1))
    score = ExponentialFamilyGLR.from_family("gaussian_mean_variance")
    detector = GridDetector(score=score, threshold=50.0)
    state = detector.init_state()
    for x in data:
        state, out = detector.update(state, x)
        assert not out["alarm"]


# ---------------------------------------------------------------------------
# gaussian_covariance — MLE and end-to-end
# ---------------------------------------------------------------------------


def test_gaussian_covariance_theta_init_shape():
    """gaussian_covariance theta_init has length p*(p+1)//2 and diagonal -0.5."""
    for p in [2, 3, 4]:
        score = ExponentialFamilyGLR.from_family("gaussian_covariance", n_features=p)
        v = p * (p + 1) // 2
        assert score._theta_init.shape == (v,)
        # diagonal entries of vech(-0.5*I) are -0.5
        idx = 0
        for i in range(p):
            for j in range(i, p):
                expected = -0.5 if i == j else 0.0
                assert np.isclose(score._theta_init[idx], expected), (
                    f"p={p}, ({i},{j}): expected {expected}, got {score._theta_init[idx]}"
                )
                idx += 1


def test_from_family_gaussian_covariance_detects_variance_shift():
    """gaussian_covariance (p=3) detects a change in the covariance matrix."""
    rng = np.random.default_rng(seed=10)
    p = 3
    sigma_pre = np.eye(p)
    sigma_post = np.diag([1.0, 4.0, 9.0])
    pre = rng.multivariate_normal(np.zeros(p), sigma_pre, 200)
    post = rng.multivariate_normal(np.zeros(p), sigma_post, 200)
    data = np.vstack([pre, post])

    score = ExponentialFamilyGLR.from_family("gaussian_covariance", n_features=p)
    detector = GridDetector(score=score, threshold=2.0)
    state = detector.init_state()
    post_shift_alarms = []
    for i, x in enumerate(data):
        state, out = detector.update(state, x)
        if i >= 200 and out["alarm"]:
            post_shift_alarms.append(i)
    assert len(post_shift_alarms) > 0


def test_from_family_gaussian_covariance_no_false_alarm():
    """gaussian_covariance (p=2) does not alarm under null with high threshold."""
    rng = np.random.default_rng(seed=11)
    p = 2
    data = rng.multivariate_normal(np.zeros(p), np.eye(p), 200)
    score = ExponentialFamilyGLR.from_family("gaussian_covariance", n_features=p)
    detector = GridDetector(score=score, threshold=50.0)
    state = detector.init_state()
    for x in data:
        state, out = detector.update(state, x)
        assert not out["alarm"]


def test_from_family_gaussian_covariance_larger_p():
    """gaussian_covariance constructs and runs for p=5 (v=15)."""
    rng = np.random.default_rng(seed=12)
    p = 5
    sigma_post = np.diag([1.0, 2.0, 4.0, 6.0, 9.0])
    pre = rng.multivariate_normal(np.zeros(p), np.eye(p), 300)
    post = rng.multivariate_normal(np.zeros(p), sigma_post, 300)
    data = np.vstack([pre, post])

    score = ExponentialFamilyGLR.from_family("gaussian_covariance", n_features=p)
    assert score.v == 15
    detector = GridDetector(score=score, threshold=5.0)
    state = detector.init_state()
    post_shift_alarms = []
    for i, x in enumerate(data):
        state, out = detector.update(state, x)
        if i >= 300 and out["alarm"]:
            post_shift_alarms.append(i)
    assert len(post_shift_alarms) > 0


# ---------------------------------------------------------------------------
# min_seg defaults — regression tests for high-dimensional correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [100, 1000])
def test_gaussian_mean_mv_min_seg_is_2(p):
    """gaussian_mean min_seg must be 2 (not p+1) so high-dim streams can be scored."""
    score = ExponentialFamilyGLR.from_family("gaussian_mean", n_features=p)
    assert score._glr_score_fn is not None
    # With min_seg=2, a short stream should produce non-zero scores.
    # If min_seg were p+1, all candidates would be skipped and max_score==0.
    rng = np.random.default_rng(seed=0)
    data = np.vstack([rng.standard_normal((50, p)), rng.standard_normal((50, p)) + 3.0])
    detector = GridDetector(score=score, threshold=1e9)  # high threshold, never alarms
    state = detector.init_state()
    max_scores = []
    for x in data:
        state, out = detector.update(state, x)
        max_scores.append(out["max_score"])
    assert max(max_scores) > 0.0, "All scores were 0 — min_seg is likely too large"


@pytest.mark.parametrize("p", [2, 3, 5])
def test_gaussian_covariance_min_seg_is_p_plus_1(p):
    """gaussian_covariance min_seg must be p+1, not p*(p+1)//2+1."""
    score = ExponentialFamilyGLR.from_family("gaussian_covariance", n_features=p)
    assert score._glr_score_fn is not None
    # Check that a stream of length 4*(p+1) produces some non-zero scores.
    rng = np.random.default_rng(seed=1)
    n = 4 * (p + 1)
    sigma_post = np.diag([float(i + 1) for i in range(p)])
    data = np.vstack(
        [
            rng.multivariate_normal(np.zeros(p), np.eye(p), n // 2),
            rng.multivariate_normal(np.zeros(p), sigma_post, n // 2),
        ]
    )
    detector = GridDetector(score=score, threshold=1e9)
    state = detector.init_state()
    max_scores = []
    for x in data:
        state, out = detector.update(state, x)
        max_scores.append(out["max_score"])
    assert max(max_scores) > 0.0, "All scores were 0 — min_seg is likely too large"
