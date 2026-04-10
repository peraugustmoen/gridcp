"""Built-in exponential-family specifications for :class:`ExponentialFamilyGLR`.

Instantiate any family via :meth:`ExponentialFamilyGLR.from_family`::

    ExponentialFamilyGLR.from_family("gaussian_mean")
    ExponentialFamilyGLR.from_family("gaussian_mean", n_features=3)
    ExponentialFamilyGLR.from_family("gaussian_covariance", n_features=3)

Implemented families
--------------------

Scalar families (v=1, n_features=1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``gaussian_mean``
    Univariate Gaussian, change in mean (known variance = 1).
    h(x) = x,  A(θ) = θ²/2,  θ ∈ ℝ.

``gaussian_variance``
    Univariate Gaussian, change in variance (known mean = 0).
    h(x) = x²,  A(θ) = −½ log(−2θ),  θ < 0.
    Canonical parametrisation: θ = −1/(2σ²).

``poisson``
    Poisson, change in rate.
    h(x) = x,  A(θ) = exp(θ),  θ ∈ ℝ.
    Canonical parametrisation: θ = log(λ).

``exponential``
    Exponential, change in rate.
    h(x) = x,  A(θ) = −log(−θ),  θ < 0.
    Canonical parametrisation: θ = −λ (so mean = −1/θ).

``bernoulli``
    Bernoulli, change in success probability.
    h(x) = x,  A(θ) = log(1 + exp(θ)),  θ ∈ ℝ.
    Canonical parametrisation: θ = log(p/(1−p)) (log-odds).

``gamma_rate``
    Gamma, change in rate (known shape k).
    h(x) = x,  A(θ) = -k * log(-θ),  θ < 0.
    Canonical parametrisation: θ = −rate (so mean = −k/θ).

Vector families (v > 1)
~~~~~~~~~~~~~~~~~~~~~~~~

``gaussian_mean``  (with n_features > 1)
    Multivariate Gaussian, change in mean vector (known covariance = I_p).
    h(x) = x,  A(θ) = ½‖θ‖²,  v = p = n_features,  θ ∈ ℝᵖ.
    Note: ∇²A(θ) = I_p is allocated per Newton step (O(p²) memory);
    for large p prefer ``MultivariateMeanIdentityCov`` or ``MeanCUSUM``.

``gaussian_mean_variance``
    Univariate Gaussian, joint change in mean and variance.
    h(x) = (x, x²),  A(θ) = −θ₀²/(4θ₁) − ½ log(−2θ₁),
    v = 2,  n_features = 1,  θ₁ < 0.
    Canonical parametrisation: θ₀ = μ/σ², θ₁ = −1/(2σ²).
    Default theta_init = (0, −0.5), corresponding to N(0, 1).

``gaussian_covariance``  (requires n_features > 1)
    Multivariate Gaussian, change in covariance matrix (known mean = 0).
    h(x) = vech(xxᵀ) with off-diagonal entries doubled,
    A(θ) = −½ log det(−2Θ),  v = p(p+1)/2,  Θ negative definite.
    Canonical parametrisation: Θ = −½ Σ⁻¹.
    Default theta_init = vech(−½ I_p), corresponding to N(0, I_p).

Internal details
----------------

Each entry in :data:`FAMILIES` is either a plain spec dict (for families whose
structure does not depend on the observation dimension) or a factory callable
(for ``gaussian_mean`` and ``gaussian_covariance``).  Factory callables take
``n_features`` and return a spec dict.

A spec dict has the following keys:

- ``v`` (int): sufficient-statistic dimension
- ``h``, ``A``: Numba-compiled callables
- ``A_prime``, ``A_dprime``: first and second derivatives of A, required when v == 1
- ``A_grad``, ``A_hess``: gradient and Hessian of A, required when v > 1
- ``theta_init``: canonical Newton starting point (float or ndarray)
"""

import math

import numba as nb
import numpy as np

# ---------------------------------------------------------------------------
# Shared matrix helpers  (used by gaussian_mean_variance and gaussian_covariance)
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _p_from_v(v):
    """Return p such that p*(p+1)//2 == v (inverse of the vech dimension)."""
    return int((-1.0 + math.sqrt(1.0 + 8.0 * v)) / 2.0)


@nb.njit(cache=True)
def _fill_sym(theta_vec, p):
    """Fill a symmetric p×p matrix from its upper-triangle vech vector."""
    Theta = np.zeros((p, p))
    idx = 0
    for i in range(p):
        for j in range(i, p):
            Theta[i, j] = theta_vec[idx]
            Theta[j, i] = theta_vec[idx]
            idx += 1
    return Theta


# ---------------------------------------------------------------------------
# Gaussian mean — scalar (v=1)
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _h_gaussian_mean(x):
    return x[0]


@nb.njit(cache=True)
def _A_gaussian_mean(theta):
    return 0.5 * theta * theta


@nb.njit(cache=True)
def _Ap_gaussian_mean(theta):
    return theta


@nb.njit(cache=True)
def _App_gaussian_mean(theta):
    return 1.0


# ---------------------------------------------------------------------------
# Gaussian mean — vector (v=p)
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _h_gaussian_mean_mv(x):
    return x.copy()


@nb.njit(cache=True)
def _A_gaussian_mean_mv(theta):
    s = 0.0
    for i in range(theta.shape[0]):
        s += theta[i] * theta[i]
    return 0.5 * s


@nb.njit(cache=True)
def _Ag_gaussian_mean_mv(theta):
    return theta.copy()


@nb.njit(cache=True)
def _Ah_gaussian_mean_mv(theta):
    return np.eye(theta.shape[0])


# ---------------------------------------------------------------------------
# Gaussian mean + variance  (v=2, scalar observations)
# theta = (theta0, theta1),  theta1 < 0
# h(x) = (x, x^2)
# A(theta) = -theta0^2 / (4*theta1) - 0.5 * log(-2*theta1)
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _h_gaussian_mean_variance(x):
    out = np.empty(2)
    out[0] = x[0]
    out[1] = x[0] * x[0]
    return out


@nb.njit(cache=True)
def _A_gaussian_mean_variance(theta):
    return -theta[0] * theta[0] / (4.0 * theta[1]) - 0.5 * math.log(-2.0 * theta[1])


@nb.njit(cache=True)
def _Ag_gaussian_mean_variance(theta):
    g = np.empty(2)
    g[0] = -theta[0] / (2.0 * theta[1])
    g[1] = theta[0] * theta[0] / (4.0 * theta[1] * theta[1]) - 1.0 / (2.0 * theta[1])
    return g


@nb.njit(cache=True)
def _Ah_gaussian_mean_variance(theta):
    H = np.zeros((2, 2))
    H[0, 0] = -1.0 / (2.0 * theta[1])
    H[0, 1] = theta[0] / (2.0 * theta[1] * theta[1])
    H[1, 0] = H[0, 1]
    H[1, 1] = -theta[0] * theta[0] / (2.0 * theta[1] ** 3) + 1.0 / (
        2.0 * theta[1] * theta[1]
    )
    return H


# ---------------------------------------------------------------------------
# Gaussian covariance  (known mean = 0, v = p*(p+1)//2, multivariate)
# theta = vech(Theta),  Theta negative definite (all eigenvalues < 0)
# h(x) = vech(x x^T) with off-diagonal entries doubled
# A(theta) = -0.5 * log det(-2 * Theta)
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _h_gaussian_covariance(x):
    p = x.shape[0]
    v = p * (p + 1) // 2
    out = np.empty(v)
    idx = 0
    for i in range(p):
        for j in range(i, p):
            if i == j:
                out[idx] = x[i] * x[j]
            else:
                out[idx] = 2.0 * x[i] * x[j]
            idx += 1
    return out


@nb.njit(cache=True)
def _A_gaussian_covariance(theta):
    p = _p_from_v(theta.shape[0])
    Theta = _fill_sym(theta, p)
    M = -2.0 * Theta
    sign, ld = np.linalg.slogdet(M)
    if sign <= 0.0:
        return np.inf
    return -0.5 * ld


@nb.njit(cache=True)
def _Ag_gaussian_covariance(theta):
    v = theta.shape[0]
    p = _p_from_v(v)
    Theta = _fill_sym(theta, p)
    M = -2.0 * Theta
    sign, _ = np.linalg.slogdet(M)
    if sign <= 0.0:
        return np.full(v, np.nan)
    Mi = np.linalg.inv(M)
    g = np.empty(v)
    idx = 0
    for i in range(p):
        for j in range(i, p):
            if i == j:
                g[idx] = Mi[i, i]
            else:
                g[idx] = 2.0 * Mi[i, j]
            idx += 1
    return g


@nb.njit(cache=True)
def _Ah_gaussian_covariance(theta):
    v = theta.shape[0]
    p = _p_from_v(v)
    Theta = _fill_sym(theta, p)
    M = -2.0 * Theta
    sign, _ = np.linalg.slogdet(M)
    if sign <= 0.0:
        return np.full((v, v), np.nan)
    Mi = np.linalg.inv(M)
    rows = np.empty(v, dtype=nb.int64)
    cols = np.empty(v, dtype=nb.int64)
    idx = 0
    for i in range(p):
        for j in range(i, p):
            rows[idx] = i
            cols[idx] = j
            idx += 1
    H = np.empty((v, v))
    for a in range(v):
        i, j = rows[a], cols[a]
        c_a = 1.0 if i == j else 2.0
        for b in range(v):
            k, m = rows[b], cols[b]
            c_b = 1.0 if k == m else 2.0
            H[a, b] = c_a * c_b * (Mi[i, k] * Mi[j, m] + Mi[i, m] * Mi[j, k])
    return H


# ---------------------------------------------------------------------------
# Poisson  (theta in R, rate = exp(theta))
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _h_poisson(x):
    return x[0]


@nb.njit(cache=True)
def _A_poisson(theta):
    return math.exp(theta)


@nb.njit(cache=True)
def _Ap_poisson(theta):
    return math.exp(theta)


@nb.njit(cache=True)
def _App_poisson(theta):
    return math.exp(theta)


# ---------------------------------------------------------------------------
# Bernoulli  (theta in R, p = sigmoid(theta))
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _h_bernoulli(x):
    return x[0]


@nb.njit(cache=True)
def _A_bernoulli(theta):
    # Numerically stable softplus: log(1 + exp(theta))
    if theta >= 0.0:
        return theta + math.log(1.0 + math.exp(-theta))
    return math.log(1.0 + math.exp(theta))


@nb.njit(cache=True)
def _Ap_bernoulli(theta):
    # sigmoid(theta) = 1 / (1 + exp(-theta))
    return 1.0 / (1.0 + math.exp(-theta))


@nb.njit(cache=True)
def _App_bernoulli(theta):
    s = 1.0 / (1.0 + math.exp(-theta))
    return s * (1.0 - s)


# ---------------------------------------------------------------------------
# Exponential  (theta < 0, rate = -theta, mean = -1/theta)
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _h_exponential(x):
    return x[0]


@nb.njit(cache=True)
def _A_exponential(theta):
    return -math.log(-theta)


@nb.njit(cache=True)
def _Ap_exponential(theta):
    return -1.0 / theta


@nb.njit(cache=True)
def _App_exponential(theta):
    return 1.0 / (theta * theta)


# ---------------------------------------------------------------------------
# Gamma rate  (theta < 0, rate = -theta, known shape k)
# h(x) = x,  A(theta) = -k * log(-theta)
# Special case k=1 is identical to Exponential.
# ---------------------------------------------------------------------------


def _gamma_rate_spec(n_features: int, *, shape: float = 1.0) -> dict:
    """Return a Gamma-rate spec with the given shape parameter.

    Parameters
    ----------
    shape : float
        Known shape parameter k > 0.  Defaults to 1.0 (Exponential).
    """
    k = float(shape)

    @nb.njit
    def _h_gr(x):
        return x[0]

    @nb.njit
    def _A_gr(theta):
        return -k * math.log(-theta)

    @nb.njit
    def _Ap_gr(theta):
        return -k / theta

    @nb.njit
    def _App_gr(theta):
        return k / (theta * theta)

    return {
        "v": 1,
        "h": _h_gr,
        "A": _A_gr,
        "A_prime": _Ap_gr,
        "A_dprime": _App_gr,
        "theta_init": -1.0,
        "theta_max": 0.0,
        "min_seg": 2,
    }


# ---------------------------------------------------------------------------
# Gaussian variance  (theta < 0, theta = -1/(2*sigma^2))
# h(x) = x^2,  A(theta) = -0.5 * log(-2*theta)
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _h_gaussian_variance(x):
    return x[0] * x[0]


@nb.njit(cache=True)
def _A_gaussian_variance(theta):
    return -0.5 * math.log(-2.0 * theta)


@nb.njit(cache=True)
def _Ap_gaussian_variance(theta):
    return -1.0 / (2.0 * theta)


@nb.njit(cache=True)
def _App_gaussian_variance(theta):
    return 1.0 / (2.0 * theta * theta)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _gaussian_mean_spec(n_features: int) -> dict:
    """Return the appropriate Gaussian-mean spec for the given observation dim."""
    if n_features == 1:
        return {
            "v": 1,
            "h": _h_gaussian_mean,
            "A": _A_gaussian_mean,
            "A_prime": _Ap_gaussian_mean,
            "A_dprime": _App_gaussian_mean,
            "theta_init": 0.0,
            "min_seg": 1,
        }
    return {
        "v": n_features,
        "h": _h_gaussian_mean_mv,
        "A": _A_gaussian_mean_mv,
        "A_grad": _Ag_gaussian_mean_mv,
        "A_hess": _Ah_gaussian_mean_mv,
        "theta_init": np.zeros(n_features, dtype=np.float64),
        "min_seg": 1,
    }


def _gaussian_covariance_spec(n_features: int) -> dict:
    """Return the Gaussian-covariance spec for p-dimensional observations.

    Parameters
    ----------
    n_features : int
        Observation dimension p.  The sufficient-statistic dimension is
        v = p*(p+1)//2.

    Notes
    -----
    The canonical theta_init corresponds to N(0, I_p): Theta = -½ I_p,
    so vech(Theta) = (-0.5, 0, 0, ..., -0.5, ...) with -0.5 on the diagonal
    entries and 0 on the off-diagonal entries.
    """
    p = n_features
    v = p * (p + 1) // 2
    Theta_init = -0.5 * np.eye(p)
    theta_init = np.array(
        [Theta_init[i, j] for i in range(p) for j in range(i, p)],
        dtype=np.float64,
    )
    return {
        "v": v,
        "h": _h_gaussian_covariance,
        "A": _A_gaussian_covariance,
        "A_grad": _Ag_gaussian_covariance,
        "A_hess": _Ah_gaussian_covariance,
        "theta_init": theta_init,
        # Need at least p+1 observations per segment for a full-rank sample
        # covariance matrix; v+1 = p(p+1)/2+1 would be far too conservative.
        "min_seg": p + 1,
    }


FAMILIES: dict = {
    "bernoulli": {
        "v": 1,
        "h": _h_bernoulli,
        "A": _A_bernoulli,
        "A_prime": _Ap_bernoulli,
        "A_dprime": _App_bernoulli,
        "theta_init": 0.0,
        "min_seg": 2,
    },
    "gamma_rate": _gamma_rate_spec,
    "exponential": {
        "v": 1,
        "h": _h_exponential,
        "A": _A_exponential,
        "A_prime": _Ap_exponential,
        "A_dprime": _App_exponential,
        "theta_init": -1.0,
        "theta_max": 0.0,
        "min_seg": 2,
    },
    "gaussian_covariance": _gaussian_covariance_spec,  # factory
    "gaussian_mean": _gaussian_mean_spec,  # factory
    "gaussian_mean_variance": {
        "v": 2,
        "h": _h_gaussian_mean_variance,
        "A": _A_gaussian_mean_variance,
        "A_grad": _Ag_gaussian_mean_variance,
        "A_hess": _Ah_gaussian_mean_variance,
        "theta_init": np.array([0.0, -0.5]),
        "min_seg": 3,
    },
    "gaussian_variance": {
        "v": 1,
        "h": _h_gaussian_variance,
        "A": _A_gaussian_variance,
        "A_prime": _Ap_gaussian_variance,
        "A_dprime": _App_gaussian_variance,
        "theta_init": -0.5,
        "theta_max": 0.0,
        "min_seg": 2,
    },
    "poisson": {
        "v": 1,
        "h": _h_poisson,
        "A": _A_poisson,
        "A_prime": _Ap_poisson,
        "A_dprime": _App_poisson,
        "theta_init": 0.0,
        "min_seg": 2,
    },
}

VALID_FAMILIES: tuple = tuple(sorted(FAMILIES.keys()))
