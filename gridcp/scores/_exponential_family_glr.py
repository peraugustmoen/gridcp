"""Generalised Likelihood Ratio (GLR) score for canonical exponential families.

Pipeline
--------
1. ``ExponentialFamilyGLR.__init__`` / ``from_family``:
   Builds a Newton MLE solver (``make_newton_solver`` for v=1,
   ``make_vector_newton_solver`` for v>1) and closes it together with the
   user-supplied callables into a GLR score function (``_make_scalar_glr_score_fn`` or
   ``_make_vector_glr_score_fn``).

2. ``update``: Accumulates ``h(x)`` into ``ExponentialFamilyGLRState``
   (running sufficient statistic + sample count).

3. ``compute_penalised_scores``: Passes the sufficient statistics of all
   O(log t) grid candidates to the pre-built GLR score function, which computes the GLR
   score for each candidate by fitting three MLEs (pre, post, null) via
   Newton's method.  Raw scores are divided by the penalty before returning.
"""

import math
from dataclasses import dataclass, field

import numba as nb
import numpy as np

from gridcp.scores._families import FAMILIES, VALID_FAMILIES
from gridcp.scores._score_helpers import as_obs
from gridcp.typing import ArrayLike, PenaltyType

# ---------------------------------------------------------------------------
# Module-level Numba helpers
# ---------------------------------------------------------------------------


@nb.njit(cache=True, inline="always")
def _norm(a):
    """Euclidean norm of a 1-D array."""
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i] * a[i]
    return math.sqrt(s)


@nb.njit(cache=True, inline="always")
def _solve(H_reg, residual):
    """Solve H_reg @ x = residual with an explicit formula for n=2."""
    n = H_reg.shape[0]
    if n == 2:
        det = H_reg[0, 0] * H_reg[1, 1] - H_reg[0, 1] * H_reg[1, 0]
        inv_det = 1.0 / det
        x0 = (H_reg[1, 1] * residual[0] - H_reg[0, 1] * residual[1]) * inv_det
        x1 = (H_reg[0, 0] * residual[1] - H_reg[1, 0] * residual[0]) * inv_det
        return np.array([x0, x1])
    else:
        return np.linalg.solve(H_reg, residual)


def _is_numba_compiled(fn):
    """Check if a function is Numba-compiled."""
    return hasattr(fn, "py_func")


# ---------------------------------------------------------------------------
# Newton solvers
# ---------------------------------------------------------------------------


def make_newton_solver(
    A_prime,
    A_dprime,
    theta_min: float = -math.inf,
    theta_max: float = math.inf,
):
    """Build a scalar Newton MLE solver for a 1D exponential family.

    Solves the MLE equation A'(θ) = S / n by Newton iterations with
    backtracking line search.  Each candidate step is rejected if it leaves
    the natural parameter domain ``(theta_min, theta_max)`` or if it does not
    reduce the residual ``|A'(θ) − S/n|``.  The step is halved up to 20
    times; if no acceptable step is found the current iterate is returned.

    Parameters
    ----------
    A_prime : callable
        First derivative of the log-partition function A.
    A_dprime : callable
        Second derivative of A (must return a positive float).
    theta_min : float, optional
        Lower bound of the natural parameter domain (exclusive).
        Defaults to ``-inf`` (unbounded below).
    theta_max : float, optional
        Upper bound of the natural parameter domain (exclusive).
        Defaults to ``+inf`` (unbounded above).

    Returns
    -------
    callable
        ``solver(S, n, theta_init, tol=1e-8, max_iter=50) -> float``
        that returns the MLE θ satisfying A'(θ) ≈ S / n.
    """
    use_numba = _is_numba_compiled(A_prime) and _is_numba_compiled(A_dprime)

    if use_numba:

        @nb.njit(cache=True)
        def solver(S, n, theta_init, tol=1e-8, max_iter=50):
            theta = theta_init
            target = S / n
            for _ in range(max_iter):
                residual = A_prime(theta) - target
                if abs(residual) < tol:
                    return theta
                adp = A_dprime(theta)
                if adp < 1e-15:
                    return theta
                step = residual / adp
                step_scale = 1.0
                accepted = False
                for _ in range(20):
                    theta_cand = theta - step_scale * step
                    if theta_cand <= theta_min or theta_cand >= theta_max:
                        step_scale *= 0.5
                        continue
                    res_cand = A_prime(theta_cand) - target
                    if math.isfinite(res_cand) and abs(res_cand) <= abs(residual):
                        theta = theta_cand
                        accepted = True
                        break
                    step_scale *= 0.5
                if not accepted:
                    return theta
            return theta

    else:

        def solver(S, n, theta_init, tol=1e-8, max_iter=50):
            theta = theta_init
            target = S / n
            for _ in range(max_iter):
                residual = A_prime(theta) - target
                if abs(residual) < tol:
                    return theta
                adp = A_dprime(theta)
                if adp < 1e-15:
                    return theta
                step = residual / adp
                step_scale = 1.0
                accepted = False
                for _ in range(20):
                    theta_cand = theta - step_scale * step
                    if theta_cand <= theta_min or theta_cand >= theta_max:
                        step_scale *= 0.5
                        continue
                    try:
                        res_cand = A_prime(theta_cand) - target
                        ok = math.isfinite(res_cand) and abs(res_cand) <= abs(residual)
                    except (ZeroDivisionError, OverflowError, ValueError):
                        ok = False
                    if ok:
                        theta = theta_cand
                        accepted = True
                        break
                    step_scale *= 0.5
                if not accepted:
                    return theta
            return theta

    return solver


def make_vector_newton_solver(A_grad, A_hess):
    """Build a vector Newton MLE solver with backtracking line search.

    Solves the MLE equation ∇A(θ) = S / n by damped Newton iterations.
    The Hessian is regularised by adding 1e-6 to the diagonal to improve
    conditioning, and a backtracking line search ensures each step reduces
    the residual norm.

    Parameters
    ----------
    A_grad : callable
        Gradient of A, mapping shape ``(v,) -> (v,)``.
    A_hess : callable
        Hessian of A, mapping shape ``(v,) -> (v, v)``.

    Returns
    -------
    callable
        ``solver(S_vec, n, theta_init, tol=1e-8, max_iter=50) -> np.ndarray``
        that returns the MLE θ satisfying ∇A(θ) ≈ S_vec / n.
    """
    use_numba = _is_numba_compiled(A_grad) and _is_numba_compiled(A_hess)

    if use_numba:

        @nb.njit(cache=True)
        def solver(S_vec, n, theta_init, tol=1e-8, max_iter=50):
            theta = theta_init.copy()
            target = S_vec / n

            grad = A_grad(theta)
            residual = grad - target
            res_norm = _norm(residual)

            for _ in range(max_iter):
                if res_norm < tol:
                    return theta
                if not np.isfinite(res_norm):
                    return theta

                H = A_hess(theta)
                h_finite = True
                for i in range(H.shape[0]):
                    if not np.isfinite(H[i, i]):
                        h_finite = False
                        break
                if not h_finite:
                    return theta

                for i in range(H.shape[0]):
                    H[i, i] += 1e-6

                step = _solve(H, residual)

                accepted = False
                step_scale = 1.0
                for _ in range(20):
                    theta_cand = theta - step_scale * step
                    grad_cand = A_grad(theta_cand)
                    residual_cand = grad_cand - target
                    res_norm_cand = _norm(residual_cand)
                    if np.isfinite(res_norm_cand) and res_norm_cand <= res_norm:
                        theta = theta_cand
                        grad = grad_cand
                        residual = residual_cand
                        res_norm = res_norm_cand
                        accepted = True
                        break
                    step_scale *= 0.5

                if not accepted:
                    return theta

            return theta

    else:

        def solver(S_vec, n, theta_init, tol=1e-8, max_iter=50):
            theta = theta_init.copy()
            target = S_vec / n

            grad = A_grad(theta)
            residual = grad - target
            res_norm = np.linalg.norm(residual)

            for _ in range(max_iter):
                if res_norm < tol:
                    return theta
                if not np.isfinite(res_norm):
                    return theta

                H = A_hess(theta)
                if not np.all(np.isfinite(np.diag(H))):
                    return theta

                H = H + 1e-6 * np.eye(H.shape[0])
                step = np.linalg.solve(H, residual)

                accepted = False
                step_scale = 1.0
                for _ in range(20):
                    theta_cand = theta - step_scale * step
                    grad_cand = A_grad(theta_cand)
                    residual_cand = grad_cand - target
                    res_norm_cand = np.linalg.norm(residual_cand)
                    if np.isfinite(res_norm_cand) and res_norm_cand <= res_norm:
                        theta = theta_cand
                        grad = grad_cand
                        residual = residual_cand
                        res_norm = res_norm_cand
                        accepted = True
                        break
                    step_scale *= 0.5

                if not accepted:
                    return theta

            return theta

    return solver


# ---------------------------------------------------------------------------
# GLR kernel factories
# ---------------------------------------------------------------------------


def _make_scalar_glr_score_fn(
    A, solver, A_prime, A_dprime, theta_init, min_seg, theta_min, theta_max
):
    """Build a GLR kernel for the scalar (v=1) case.

    Parameters
    ----------
    A : callable
        Log-partition function, ``float -> float``.
    solver : callable
        Scalar Newton MLE solver, as produced by ``make_newton_solver``.
    A_prime : callable
        First derivative of A.
    A_dprime : callable
        Second derivative of A.
    theta_init : float
        Starting point for the Newton MLE solver.
    min_seg : int
        Minimum number of observations required on each side of a candidate
        changepoint; candidates with fewer are assigned a score of 0.
    theta_min : float
        Lower bound of the natural parameter domain (exclusive).
    theta_max : float
        Upper bound of the natural parameter domain (exclusive).

    Returns
    -------
    callable
        ``kernel(total_stat, before_stats, t, before_n) -> np.ndarray``
        returning raw (unpenalised) GLR scores for all candidates.
    """
    _eps = 1e-10  # small buffer to keep warm-starts strictly inside the domain
    _lo = theta_min + _eps if math.isfinite(theta_min) else theta_min
    _hi = theta_max - _eps if math.isfinite(theta_max) else theta_max

    use_numba = (
        _is_numba_compiled(A)
        and _is_numba_compiled(solver)
        and _is_numba_compiled(A_prime)
        and _is_numba_compiled(A_dprime)
    )

    if use_numba:

        @nb.njit(cache=True)
        def kernel(total_stat, before_stats, t, before_n):
            n_cand = before_n.shape[0]
            out = np.zeros(n_cand, dtype=np.float64)
            S_total = total_stat[0]
            adp0 = A_dprime(theta_init)
            if adp0 < 1e-15:
                adp0 = 1e-15
            A_prime_init = A_prime(theta_init)

            for i in range(n_cand):
                n_pre = before_n[i]
                n_post = t - n_pre
                if n_pre < min_seg or n_post < min_seg:
                    continue

                S_pre = before_stats[i, 0]
                S_post = S_total - S_pre

                # Warm-start: one Newton step from theta_init, clamped to domain
                warm_pre = theta_init + (S_pre / n_pre - A_prime_init) / adp0
                warm_post = theta_init + (S_post / n_post - A_prime_init) / adp0
                warm_null = theta_init + (S_total / t - A_prime_init) / adp0
                if warm_pre < _lo:
                    warm_pre = _lo
                elif warm_pre > _hi:
                    warm_pre = _hi
                if warm_post < _lo:
                    warm_post = _lo
                elif warm_post > _hi:
                    warm_post = _hi
                if warm_null < _lo:
                    warm_null = _lo
                elif warm_null > _hi:
                    warm_null = _hi

                th_pre = solver(S_pre, n_pre, warm_pre)
                th_post = solver(S_post, n_post, warm_post)
                th_null = solver(S_total, t, warm_null)

                ell_pre = th_pre * S_pre - n_pre * A(th_pre)
                ell_post = th_post * S_post - n_post * A(th_post)
                ell_null = th_null * S_total - t * A(th_null)

                glr = ell_pre + ell_post - ell_null
                out[i] = glr if glr > 0.0 else 0.0

            return out

    else:

        def kernel(total_stat, before_stats, t, before_n):
            n_cand = before_n.shape[0]
            out = np.zeros(n_cand, dtype=np.float64)
            S_total = total_stat[0]
            adp0 = A_dprime(theta_init)
            if adp0 < 1e-15:
                adp0 = 1e-15
            A_prime_init = A_prime(theta_init)

            for i in range(n_cand):
                n_pre = before_n[i]
                n_post = t - n_pre
                if n_pre < min_seg or n_post < min_seg:
                    continue

                S_pre = before_stats[i, 0]
                S_post = S_total - S_pre

                # Warm-start: one Newton step from theta_init, clamped to domain
                warm_pre = theta_init + (S_pre / n_pre - A_prime_init) / adp0
                warm_post = theta_init + (S_post / n_post - A_prime_init) / adp0
                warm_null = theta_init + (S_total / t - A_prime_init) / adp0
                warm_pre = max(_lo, min(_hi, warm_pre))
                warm_post = max(_lo, min(_hi, warm_post))
                warm_null = max(_lo, min(_hi, warm_null))

                th_pre = solver(S_pre, n_pre, warm_pre)
                th_post = solver(S_post, n_post, warm_post)
                th_null = solver(S_total, t, warm_null)

                ell_pre = th_pre * S_pre - n_pre * A(th_pre)
                ell_post = th_post * S_post - n_post * A(th_post)
                ell_null = th_null * S_total - t * A(th_null)

                glr = ell_pre + ell_post - ell_null
                out[i] = glr if glr > 0.0 else 0.0

            return out

    return kernel


def _make_vector_glr_score_fn(A, solver, A_grad, A_hess, theta_init, min_seg):
    """Build a GLR kernel for the vector (v > 1) case.

    Parameters
    ----------
    A : callable
        Log-partition function, ``1D array of length v -> float``.
    solver : callable
        Vector Newton MLE solver, as produced by ``make_vector_newton_solver``.
    A_grad : callable
        Gradient of A, shape ``(v,) -> (v,)``.
    A_hess : callable
        Hessian of A, shape ``(v,) -> (v, v)``.
    theta_init : np.ndarray, shape (v,)
        Starting point for the Newton MLE solver.
    min_seg : int
        Minimum segment length; candidates with fewer observations on either
        side are assigned a score of 0.

    Returns
    -------
    callable
        ``kernel(total_stat, before_stats, t, before_n) -> np.ndarray``
        returning raw (unpenalised) GLR scores for all candidates.
    """
    use_numba = (
        _is_numba_compiled(A)
        and _is_numba_compiled(solver)
        and _is_numba_compiled(A_grad)
        and _is_numba_compiled(A_hess)
    )

    if use_numba:

        @nb.njit(cache=True)
        def _log_lik_vec(theta, S, n):
            result = 0.0
            for k in range(theta.shape[0]):
                result += theta[k] * S[k]
            return result - n * A(theta)

        @nb.njit(cache=True)
        def kernel(total_stat, before_stats, t, before_n):
            n_cand = before_n.shape[0]
            out = np.zeros(n_cand, dtype=np.float64)

            for i in range(n_cand):
                n_pre = before_n[i]
                n_post = t - n_pre
                if n_pre < min_seg or n_post < min_seg:
                    continue

                S_pre = before_stats[i].copy()
                S_post = total_stat - S_pre

                th_pre = solver(S_pre, n_pre, theta_init.copy())
                th_post = solver(S_post, n_post, theta_init.copy())
                th_null = solver(total_stat, t, theta_init.copy())

                glr = (
                    _log_lik_vec(th_pre, S_pre, n_pre)
                    + _log_lik_vec(th_post, S_post, n_post)
                    - _log_lik_vec(th_null, total_stat, t)
                )
                out[i] = glr if glr > 0.0 else 0.0

            return out

    else:

        def _log_lik_vec(theta, S, n):
            return np.dot(theta, S) - n * A(theta)

        def kernel(total_stat, before_stats, t, before_n):
            n_cand = before_n.shape[0]
            out = np.zeros(n_cand, dtype=np.float64)

            for i in range(n_cand):
                n_pre = before_n[i]
                n_post = t - n_pre
                if n_pre < min_seg or n_post < min_seg:
                    continue

                S_pre = before_stats[i].copy()
                S_post = total_stat - S_pre

                th_pre = solver(S_pre, n_pre, theta_init.copy())
                th_post = solver(S_post, n_post, theta_init.copy())
                th_null = solver(total_stat, t, theta_init.copy())

                glr = (
                    _log_lik_vec(th_pre, S_pre, n_pre)
                    + _log_lik_vec(th_post, S_post, n_post)
                    - _log_lik_vec(th_null, total_stat, t)
                )
                out[i] = glr if glr > 0.0 else 0.0

            return out

    return kernel


# ---------------------------------------------------------------------------
# State and Score class
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExponentialFamilyGLRState:
    """Running sufficient statistics for the exponential family GLR score.

    Attributes
    ----------
    n_samples : int
        Number of observations incorporated so far.
    suff_stat : np.ndarray, shape (v,)
        Cumulative sum of h(xᵢ) over all observations seen, where h is the
        sufficient-statistic function of the exponential family.
    """

    n_samples: int = 0
    suff_stat: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


class ExponentialFamilyGLR:
    """GLR score for canonical exponential families.

    Canonical exponential families have density of the form

        f(x; θ) = c(x) exp(θᵀ h(x) - A(θ)).

    Here, θ is the natural parameter, h(x) is the sufficient statistic, and
    A(θ) is the log-partition function. c(x) is the base measure, which does not depend on θ and
    is thus not needed to compute the GLR score.

    The user supplies the sufficient statistic ``h``, log-partition ``A``, and
    first and second derivatives of A. Built-in families are available via :meth:`from_family`.

    The default centering/penalty calibration is asymptotic: it uses the
    centered statistic ``2*GLR - v`` together with the Wilks-style penalty
    ``sqrt(v log t) + log t``. This is not exact finite-sample.

    Parameters
    ----------
    v : int
        Dimension of the sufficient statistic h(x).  Use ``v=1`` for
        one-parameter families (Gaussian mean, Poisson, Bernoulli, Exponential)
        and ``v>1`` for multi-parameter families (e.g. Gaussian mean+variance).
    h : callable
        Sufficient-statistic function.  Accepts a 1D ``float64`` array of
        length ``n_features`` and returns a ``float`` when ``v=1``, or a 1D
        array of length ``v`` when ``v>1``.  May be a Numba-compiled or plain
        NumPy callable.
    A : callable
        Log-partition function.  ``float -> float`` for ``v=1``;
        ``1D array of length v -> float`` for ``v>1``.
    n_features : int
        Dimension of each observation vector.
    A_prime : callable, optional
        First derivative A'(θ).  Required when ``v=1``; ignored otherwise.
    A_dprime : callable, optional
        Second derivative A''(θ).  Required when ``v=1``; ignored otherwise.
    A_grad : callable, optional
        Gradient ∇A(θ), shape ``(v,) -> (v,)``.  Required when ``v>1``;
        ignored otherwise.
    A_hess : callable, optional
        Hessian ∇²A(θ), shape ``(v,) -> (v, v)``.  Required when ``v>1``;
        ignored otherwise.
    theta_init : float or np.ndarray, optional
        Starting point for the Newton MLE solver.  Defaults to ``0.0``
        (scalar) or ``np.zeros(v)`` (vector).
    theta_min : float, optional
        Exclusive lower bound of the natural parameter domain.  The Newton
        solver will never propose a candidate ≤ ``theta_min``.  Defaults to
        ``-inf`` (unbounded).  Only used when ``v=1``.
    theta_max : float, optional
        Exclusive upper bound of the natural parameter domain.  The Newton
        solver will never propose a candidate ≥ ``theta_max``.  Defaults to
        ``+inf`` (unbounded).  Only used when ``v=1``.
    min_seg : int or None, optional
        Minimum number of observations required on each side of a candidate
        changepoint.  Candidates with fewer observations are assigned a score
        of 0.  Defaults to ``v + 1``, which is the minimum needed to identify
        ``v`` parameters.  Must be at least 2.
    penalty : PenaltyType, optional
        Penalty type.  ``PenaltyType.TIME_DEPENDENT`` (default) uses
        ``sqrt(v log t) + log t`` after centering the ``2*GLR`` statistic by
        ``v``; ``PenaltyType.CONSTANT`` uses 1.

    Notes
    -----
    The score computed for each candidate changepoint s at time t is:

        score(s, t) = (2 GLR(s, t) - v) / penalty(t)

    where the generalised log-likelihood ratio is:

        GLR(s, t) = ℓ(θ̂_pre; x_{1:s}) + ℓ(θ̂_post; x_{s+1:t}) − ℓ(θ̂_null; x_{1:t})

    and the penalty is:

        penalty(t) = √(v log t) + log t

    A candidate triggers an alarm when its score exceeds the calibrated
    threshold.  MLEs are computed by Newton's method, warm-started one step
    from ``theta_init``.  If all supplied callables are Numba-compiled, the
    solvers and kernels are JIT-compiled for maximum performance; otherwise
    they fall back to plain NumPy.

    Examples
    --------
    Scalar Gaussian mean model (known variance = 1):

    >>> import numba as nb
    >>> import numpy as np
    >>> @nb.njit
    ... def h(x):
    ...     return x[0]
    >>> @nb.njit
    ... def A(theta):
    ...     return 0.5 * theta * theta
    >>> @nb.njit
    ... def Ap(theta):
    ...     return theta
    >>> @nb.njit
    ... def App(theta):
    ...     return 1.0
    >>> score = ExponentialFamilyGLR(
    ...     v=1, n_features=1, h=h, A=A, A_prime=Ap, A_dprime=App,
    ... )
    """

    def __init__(
        self,
        v: int,
        h,
        A,
        n_features: int,
        *,
        A_prime=None,
        A_dprime=None,
        A_grad=None,
        A_hess=None,
        theta_init=None,
        theta_min: float = -math.inf,
        theta_max: float = math.inf,
        min_seg: int | None = None,
        penalty: PenaltyType = PenaltyType.TIME_DEPENDENT,
    ):
        self.v = v
        self.n_features = n_features
        self._h = h

        # --- Defaults ---
        if theta_init is None:
            theta_init = 0.0 if v == 1 else np.zeros(v, dtype=np.float64)
        if v > 1 and not isinstance(theta_init, np.ndarray):
            theta_init = np.asarray(theta_init, dtype=np.float64)

        # --- Minimum segment length ---
        if min_seg is None:
            min_seg = v + 1
        if min_seg < 2:
            raise ValueError(f"min_seg must be >= 2, got {min_seg}.")

        # --- Build solver and GLR kernel ---
        if v == 1:
            if A_prime is None or A_dprime is None:
                raise ValueError(
                    "A_prime and A_dprime are required for scalar case (v=1)."
                )
            solver = make_newton_solver(A_prime, A_dprime, theta_min, theta_max)
            self._glr_score_fn = _make_scalar_glr_score_fn(
                A, solver, A_prime, A_dprime, theta_init, min_seg, theta_min, theta_max
            )
        else:
            if A_grad is None or A_hess is None:
                raise ValueError(
                    "A_grad and A_hess are required for vector case (v>1)."
                )
            solver = make_vector_newton_solver(A_grad, A_hess)
            self._glr_score_fn = _make_vector_glr_score_fn(
                A, solver, A_grad, A_hess, theta_init, min_seg
            )

        self.penalty = penalty
        self._theta_init = theta_init

    @classmethod
    def from_family(
        cls,
        family: str,
        n_features: int = 1,
        *,
        theta_init=None,
        min_seg: int | None = None,
        penalty: PenaltyType = PenaltyType.TIME_DEPENDENT,
        **family_kwargs,
    ) -> "ExponentialFamilyGLR":
        """Construct a GLR score for a built-in exponential family.

        Parameters
        ----------
        family : str
            Name of the exponential family.  Valid choices are:

            - ``'gaussian_mean'`` — Gaussian mean (known variance = 1);
              scalar (v=1) when ``n_features=1``, multivariate (v=p) when
              ``n_features>1``.  For large p, prefer
              ``MultivariateMeanIdentityCov`` or ``MeanCUSUM``, which use
              closed-form O(p) scores for the same model.
            - ``'gaussian_variance'`` — Gaussian variance (known mean = 0);
              scalar (v=1).
            - ``'gaussian_mean_variance'`` — joint Gaussian mean and
              variance; v=2, ``n_features=1``.
            - ``'gaussian_covariance'`` — multivariate Gaussian covariance
              (known mean = 0); v=p(p+1)/2, requires ``n_features>1``.
            - ``'poisson'`` — Poisson rate; scalar (v=1).
            - ``'exponential'`` — Exponential rate; scalar (v=1).
            - ``'bernoulli'`` — Bernoulli probability; scalar (v=1).
            - ``'gamma_rate'`` — Gamma rate (known shape k); scalar (v=1).
              Pass ``shape=k`` via ``**family_kwargs``, e.g.
              ``from_family("gamma_rate", shape=2.0)``.
              Defaults to ``shape=1.0`` (equivalent to Exponential).
        n_features : int, optional
            Dimension of each observation vector.  For ``'gaussian_mean'``
            and ``'gaussian_covariance'`` this controls the vector
            dimension.  All other families are scalar (``n_features=1``)
            regardless of this parameter.
        theta_init : float or np.ndarray, optional
            Override the family's default Newton starting point.  If
            ``None`` (default), the canonical starting point is used.
        min_seg : int or None, optional
            Passed through to ``__init__``.  Defaults to ``v + 1``.
        penalty : PenaltyType, optional
            Passed through to ``__init__``.

        Returns
        -------
        ExponentialFamilyGLR
            Fully constructed score object.

        Raises
        ------
        ValueError
            If ``family`` is not a recognised name.

        Examples
        --------
        >>> score = ExponentialFamilyGLR.from_family("gaussian_mean")
        >>> score = ExponentialFamilyGLR.from_family("gaussian_mean", n_features=3)
        >>> score = ExponentialFamilyGLR.from_family("gaussian_variance")
        >>> score = ExponentialFamilyGLR.from_family("gaussian_mean_variance")
        >>> score = ExponentialFamilyGLR.from_family("gaussian_covariance", n_features=3)
        >>> score = ExponentialFamilyGLR.from_family("poisson")
        >>> score = ExponentialFamilyGLR.from_family("bernoulli")
        >>> score = ExponentialFamilyGLR.from_family("exponential")
        >>> score = ExponentialFamilyGLR.from_family("gamma_rate", shape=2.0)
        """
        if family not in FAMILIES:
            raise ValueError(
                f"Unknown family {family!r}. "
                f"Valid choices are: {', '.join(VALID_FAMILIES)}."
            )
        spec = FAMILIES[family]
        if callable(spec) and not hasattr(spec, "py_func"):
            spec = spec(n_features, **family_kwargs)
        effective_theta_init = (
            theta_init if theta_init is not None else spec["theta_init"]
        )
        effective_n_features = n_features if spec["v"] > 1 else 1
        effective_min_seg = min_seg if min_seg is not None else spec.get("min_seg")
        kwargs: dict = dict(
            v=spec["v"],
            n_features=effective_n_features,
            h=spec["h"],
            A=spec["A"],
            theta_init=effective_theta_init,
            theta_min=spec.get("theta_min", -math.inf),
            theta_max=spec.get("theta_max", math.inf),
            min_seg=effective_min_seg,
            penalty=penalty,
        )
        if spec["v"] == 1:
            kwargs["A_prime"] = spec["A_prime"]
            kwargs["A_dprime"] = spec["A_dprime"]
        else:
            kwargs["A_grad"] = spec["A_grad"]
            kwargs["A_hess"] = spec["A_hess"]
        return cls(**kwargs)

    def _get_penalty(self, n_samples: int) -> float:
        """Return the penalty divisor for the current sample size."""
        if self.penalty == PenaltyType.TIME_DEPENDENT:
            log_t = np.log(n_samples)
            return np.sqrt(self.v * log_t) + log_t
        return 1.0

    def init_state(self) -> ExponentialFamilyGLRState:
        """Return a fresh initial state with no observations seen."""
        return ExponentialFamilyGLRState(
            n_samples=0,
            suff_stat=np.zeros(self.v, dtype=np.float64),
        )

    def update(
        self,
        state: ExponentialFamilyGLRState,
        x: ArrayLike,
    ) -> ExponentialFamilyGLRState:
        """Incorporate a new observation into the running sufficient statistic.

        Parameters
        ----------
        state : ExponentialFamilyGLRState
            Current state.
        x : ArrayLike
            New observation, shape ``(n_features,)``. Scalars are also
            accepted and will be broadcast to the expected shape.

        Returns
        -------
        ExponentialFamilyGLRState
            Updated state with ``n_samples`` incremented by 1 and
            ``suff_stat`` increased by ``h(x)``.

        Raises
        ------
        ValueError
            If the observation size does not match ``n_features``.
        """
        x_arr = as_obs(x, self.n_features)

        h_x = self._h(x_arr)
        h_arr = np.asarray(h_x, dtype=np.float64).reshape(-1)

        return ExponentialFamilyGLRState(
            n_samples=state.n_samples + 1,
            suff_stat=state.suff_stat + h_arr,
        )

    def compute_penalised_scores(
        self,
        state: ExponentialFamilyGLRState,
        grid_states: list[ExponentialFamilyGLRState],
    ) -> np.ndarray:
        """Compute a penalised GLR score for every active grid candidate.

        Parameters
        ----------
        state : ExponentialFamilyGLRState
            Global running state after the latest observation.
        grid_states : list[ExponentialFamilyGLRState]
            Per-candidate state snapshots, one per active grid point.

        Returns
        -------
        np.ndarray, shape (len(grid_states),)
            Penalised centered ``2*GLR`` score for each active candidate.
            Scores can be negative; candidates with too few observations on
            either side receive a score of 0.
        """
        before_stats = np.stack([gs.suff_stat for gs in grid_states])
        before_n = np.array([gs.n_samples for gs in grid_states], dtype=np.int64)

        raw_scores = 2.0 * self._glr_score_fn(
            state.suff_stat, before_stats, state.n_samples, before_n
        )
        centered_scores = raw_scores - self.v
        penalty = self._get_penalty(state.n_samples)
        return centered_scores / penalty
