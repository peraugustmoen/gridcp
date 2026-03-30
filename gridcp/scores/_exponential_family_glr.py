"""Generalised Likelihood Ratio (GLR) score for canonical exponential families.

Supports any exponential family whose log-partition function A has known first
and second derivatives (or gradient and Hessian for v > 1) that can be supplied
as Numba-compiled callables.  The user provides the sufficient statistic h, the
log-partition A, and its derivatives; the module builds JIT-compiled Newton MLE
solvers and GLR kernels at construction time.
"""

import math
from dataclasses import dataclass

import numba as nb
import numpy as np

from gridcp.typing import ArrayLike

# ---------------------------------------------------------------------------
# Module-level Numba helpers
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _p_from_v(v):
    """Recover p from vech dimension v = p*(p+1)/2."""
    return int((-1.0 + (1.0 + 8.0 * v) ** 0.5) / 2.0)


@nb.njit(cache=True)
def _fill_sym(theta_vec, p):
    """Reconstruct symmetric matrix from upper-triangle (row-major) vector."""
    Theta = np.zeros((p, p))
    idx = 0
    for i in range(p):
        for j in range(i, p):
            Theta[i, j] = theta_vec[idx]
            Theta[j, i] = theta_vec[idx]
            idx += 1
    return Theta


@nb.njit(cache=True, inline="always")
def _norm(a):
    """Euclidean norm of a 1-D array."""
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i] * a[i]
    return math.sqrt(s)


@nb.njit(cache=True, inline="always")
def _solve(H_reg, residual):
    """Solve H_reg @ x = residual with explicit formulas for n<=3."""
    n = H_reg.shape[0]
    if n == 2:
        det = H_reg[0, 0] * H_reg[1, 1] - H_reg[0, 1] * H_reg[1, 0]
        inv_det = 1.0 / det
        x0 = (H_reg[1, 1] * residual[0] - H_reg[0, 1] * residual[1]) * inv_det
        x1 = (H_reg[0, 0] * residual[1] - H_reg[1, 0] * residual[0]) * inv_det
        return np.array([x0, x1])
    elif n == 3:
        a, b, c = H_reg[0, 0], H_reg[0, 1], H_reg[0, 2]
        d, e, f = H_reg[1, 0], H_reg[1, 1], H_reg[1, 2]
        g, h, k = H_reg[2, 0], H_reg[2, 1], H_reg[2, 2]
        det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)
        inv_det = 1.0 / det
        x0 = (
            (e * k - f * h) * residual[0]
            + (c * h - b * k) * residual[1]
            + (b * f - c * e) * residual[2]
        ) * inv_det
        x1 = (
            (f * g - d * k) * residual[0]
            + (a * k - c * g) * residual[1]
            + (c * d - a * f) * residual[2]
        ) * inv_det
        x2 = (
            (d * h - e * g) * residual[0]
            + (b * g - a * h) * residual[1]
            + (a * e - b * d) * residual[2]
        ) * inv_det
        return np.array([x0, x1, x2])
    else:
        return np.linalg.solve(H_reg, residual)


# ---------------------------------------------------------------------------
# Scalar Newton solver (closure-based)
# ---------------------------------------------------------------------------


def make_newton_solver(A_prime, A_dprime):
    """Build a JIT-compiled scalar Newton MLE solver for a 1D exponential family.

    Solves the MLE equation A'(θ) = S / n by Newton iterations.  A domain
    guard prevents the iterate from crossing zero when ``theta_init`` is
    non-zero, which is necessary for families whose natural parameter domain
    excludes zero (e.g. Gaussian-variance with θ < 0, Exponential with θ < 0).

    Parameters
    ----------
    A_prime : callable (@nb.njit)
        First derivative of the log-partition function A.
    A_dprime : callable (@nb.njit)
        Second derivative of A (must return a positive float).

    Returns
    -------
    callable
        A ``@nb.njit``-compiled function
        ``solver(S, n, theta_init, tol=1e-8, max_iter=50) -> float``
        that returns the MLE θ satisfying A'(θ) ≈ S / n.
    """

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
            theta_new = theta - residual / adp
            # Domain guard: prevent crossing zero for log-barrier-type families
            if theta_init < 0.0 and theta_new >= 0.0:
                theta_new = theta * 0.1
            elif theta_init > 0.0 and theta_new <= 0.0:
                theta_new = theta * 0.1
            theta = theta_new
        return theta

    return solver


# ---------------------------------------------------------------------------
# Vector Newton solver (robust, with backtracking)
# ---------------------------------------------------------------------------


def make_vector_newton_solver(A_grad, A_hess):
    """Build a JIT-compiled vector Newton MLE solver with backtracking line search.

    Solves the MLE equation ∇A(θ) = S / n by damped Newton iterations.
    The Hessian is regularised by adding 1e-6 to the diagonal to improve
    conditioning, and a backtracking line search ensures each step reduces
    the residual norm.

    Parameters
    ----------
    A_grad : callable (@nb.njit)
        Gradient of A, mapping shape ``(v,) -> (v,)``.
    A_hess : callable (@nb.njit)
        Hessian of A, mapping shape ``(v,) -> (v, v)``.

    Returns
    -------
    callable
        A ``@nb.njit``-compiled function
        ``solver(S_vec, n, theta_init, tol=1e-8, max_iter=50) -> np.ndarray``
        that returns the MLE θ satisfying ∇A(θ) ≈ S_vec / n.
    """

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

            # Backtracking line search
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

    return solver


# ---------------------------------------------------------------------------
# GLR kernel factories
# ---------------------------------------------------------------------------


def _make_scalar_glr_kernel(A, solver, A_prime, A_dprime, theta_init, min_seg):
    """Build a JIT-compiled GLR kernel for the scalar (v=1) case.

    Parameters
    ----------
    A : callable (@nb.njit)
        Log-partition function, ``float -> float``.
    solver : callable (@nb.njit)
        Scalar Newton MLE solver, as produced by ``make_newton_solver``.
    A_prime : callable (@nb.njit)
        First derivative of A.
    A_dprime : callable (@nb.njit)
        Second derivative of A.
    theta_init : float
        Initial parameter value used for warm-starting the Newton solver.
    min_seg : int
        Minimum number of observations required on each side of a candidate
        changepoint; candidates with fewer are assigned a score of 0.

    Returns
    -------
    callable
        A ``@nb.njit``-compiled function
        ``kernel(total_stat, before_stats, t, before_n) -> np.ndarray``
        returning raw (unpenalised) GLR scores for all candidates.
    """

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

            # Warm-start: one Newton step from theta_init
            warm_pre = theta_init + (S_pre / n_pre - A_prime_init) / adp0
            warm_post = theta_init + (S_post / n_post - A_prime_init) / adp0
            warm_null = theta_init + (S_total / t - A_prime_init) / adp0

            # Domain guard: keep warm-start on same side of zero as theta_init
            if theta_init < 0.0:
                if warm_pre >= 0.0:
                    warm_pre = theta_init * 0.1
                if warm_post >= 0.0:
                    warm_post = theta_init * 0.1
                if warm_null >= 0.0:
                    warm_null = theta_init * 0.1
            elif theta_init > 0.0:
                if warm_pre <= 0.0:
                    warm_pre = theta_init * 0.1
                if warm_post <= 0.0:
                    warm_post = theta_init * 0.1
                if warm_null <= 0.0:
                    warm_null = theta_init * 0.1

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


def _make_vector_glr_kernel(
    A, solver, A_grad, A_hess, theta_init, min_seg, is_cov_case
):
    """Build a JIT-compiled GLR kernel for the vector (v > 1) case.

    Parameters
    ----------
    A : callable (@nb.njit)
        Log-partition function, ``1D array of length v -> float``.
    solver : callable (@nb.njit)
        Vector Newton MLE solver, as produced by ``make_vector_newton_solver``.
    A_grad : callable (@nb.njit)
        Gradient of A, shape ``(v,) -> (v,)``.
    A_hess : callable (@nb.njit)
        Hessian of A, shape ``(v,) -> (v, v)``.
    theta_init : np.ndarray, shape (v,)
        Initial parameter vector for warm-starting (used when not
        ``is_cov_case``).
    min_seg : int
        Minimum segment length; candidates with fewer observations on either
        side are assigned a score of 0.
    is_cov_case : bool
        If ``True``, use the covariance warm-start heuristic
        (θ = vech(−½ Σ̂⁻¹)) instead of the fixed ``theta_init``.

    Returns
    -------
    callable
        A ``@nb.njit``-compiled function
        ``kernel(total_stat, before_stats, t, before_n) -> np.ndarray``
        returning raw (unpenalised) GLR scores for all candidates.
    """

    @nb.njit(cache=True)
    def _log_lik_vec(theta, S, n):
        result = 0.0
        for k in range(theta.shape[0]):
            result += theta[k] * S[k]
        return result - n * A(theta)

    @nb.njit(cache=True)
    def _warm_start_cov(S, n):
        """Warm-start for covariance case: theta = vech(-0.5 * Sigma_hat^{-1})."""
        v = S.shape[0]
        p = _p_from_v(v)
        S_outer = _fill_sym(S, p)
        Sigma_hat = S_outer / n
        # _fill_sym puts the doubled off-diag h-values into both (i,j) and
        # (j,i), so halve the off-diagonal to recover the true sample cov.
        for _i in range(p):
            for _j in range(p):
                if _i != _j:
                    Sigma_hat[_i, _j] *= 0.5
        sign, _ = np.linalg.slogdet(Sigma_hat)
        if sign <= 0.0:
            return theta_init.copy()
        Theta_hat = -0.5 * np.linalg.inv(Sigma_hat)
        out = np.empty(v)
        idx = 0
        for ii in range(p):
            for jj in range(ii, p):
                out[idx] = Theta_hat[ii, jj]
                idx += 1
        return out

    @nb.njit(cache=True)
    def kernel(total_stat, before_stats, t, before_n):
        """Compute raw (unpenalised) GLR scores for all candidates."""
        n_cand = before_n.shape[0]
        out = np.zeros(n_cand, dtype=np.float64)

        for i in range(n_cand):
            n_pre = before_n[i]
            n_post = t - n_pre
            if n_pre < min_seg or n_post < min_seg:
                continue

            S_pre = before_stats[i].copy()
            S_post = total_stat - S_pre
            S_null = total_stat

            if is_cov_case:
                warm_pre = _warm_start_cov(S_pre, n_pre)
                warm_post = _warm_start_cov(S_post, n_post)
                warm_null = _warm_start_cov(S_null, t)
            else:
                warm_pre = theta_init.copy()
                warm_post = theta_init.copy()
                warm_null = theta_init.copy()

            th_pre = solver(S_pre, n_pre, warm_pre)
            th_post = solver(S_post, n_post, warm_post)
            th_null = solver(S_null, t, warm_null)

            glr = (
                _log_lik_vec(th_pre, S_pre, n_pre)
                + _log_lik_vec(th_post, S_post, n_post)
                - _log_lik_vec(th_null, S_null, t)
            )

            out[i] = glr if glr > 0.0 else 0.0

        return out

    return kernel


# ---------------------------------------------------------------------------
# Penalty factory
# ---------------------------------------------------------------------------


def _make_penalty(v):
    """Build a JIT-compiled penalty function for GLR statistics.

    The penalty grows as ``v * (log t + sqrt(log t))``, which ensures
    asymptotic ARL control under the null.

    Parameters
    ----------
    v : int
        Sufficient-statistic dimension.  Scales the penalty linearly.

    Returns
    -------
    callable
        A ``@nb.njit``-compiled function ``penalty(t) -> float``,
        where ``t`` is the current number of observations.
    """

    @nb.njit(cache=True)
    def penalty(t):
        log_t = np.log(t)
        return v * (log_t + np.sqrt(log_t))

    return penalty


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
    suff_stat: np.ndarray = None  # shape (v,); set by ExponentialFamilyGLR.init_state


class ExponentialFamilyGLR:
    """General GLR score for canonical exponential families.

    Implements the ``ScoreModel`` protocol from ``gridcp.typing`` so it can
    be used directly with ``GridDetector``.

    The user provides the exponential-family specification (``h``, ``A``, and
    its derivatives) and the constructor builds Numba-compiled MLE solvers and
    GLR kernels once at construction time.  All subsequent per-step computations
    run at full JIT speed with no Python callback overhead.

    Parameters
    ----------
    v : int
        Dimension of the sufficient statistic h(x).  Use ``v=1`` for
        one-parameter families (Gaussian mean, Poisson, Bernoulli, Exponential)
        and ``v>1`` for multi-parameter families (e.g. Gaussian mean+variance).
    h : callable (@nb.njit)
        Sufficient-statistic function.  Accepts a 1D ``float64`` array of
        length ``n_features`` and returns a ``float`` when ``v=1``, or a 1D
        array of length ``v`` when ``v>1``.
    A : callable (@nb.njit)
        Log-partition function.  ``float -> float`` for ``v=1``;
        ``1D array of length v -> float`` for ``v>1``.
    n_features : int
        Dimension of each observation vector.
    A_prime : callable (@nb.njit), optional
        First derivative A'(θ).  Required when ``v=1``; ignored otherwise.
    A_dprime : callable (@nb.njit), optional
        Second derivative A''(θ).  Required when ``v=1``; ignored otherwise.
    A_grad : callable (@nb.njit), optional
        Gradient ∇A(θ), shape ``(v,) -> (v,)``.  Required when ``v>1``;
        ignored otherwise.
    A_hess : callable (@nb.njit), optional
        Hessian ∇²A(θ), shape ``(v,) -> (v, v)``.  Required when ``v>1``;
        ignored otherwise.
    theta_init : float or np.ndarray, optional
        Starting point for the Newton MLE solver.  Defaults to ``0.0``
        (scalar) or ``np.zeros(v)`` (vector).  For families whose natural
        parameter domain excludes zero (e.g. ``θ < 0`` for Gaussian-variance
        or Exponential), you **must** supply a ``theta_init`` on the correct
        side; the Newton solver applies a domain guard to keep iterates there.
    min_seg : int or None, optional
        Minimum number of observations required on each side of a candidate
        changepoint.  Candidates with fewer observations are assigned a score
        of 0.  Defaults to ``v + 1``, which is the minimum needed to identify
        ``v`` parameters.  Must be at least 2.
    cov_parametrization : bool, optional
        Set to ``True`` when the sufficient statistic ``h`` returns the
        half-vectorisation (vech) of a covariance-like outer product, i.e.
        ``v = p*(p+1)/2`` with ``p = n_features``.  When enabled, the Newton
        solver is warm-started from the sample-covariance inverse
        ``θ = vech(−½ Σ̂⁻¹)`` instead of the fixed ``theta_init``, which
        greatly improves convergence.  Default is ``False``.

    Notes
    -----
    The score computed for each candidate changepoint s at time t is:

        score(s, t) = GLR(s, t) / penalty(t)

    where the generalised log-likelihood ratio is:

        GLR(s, t) = ℓ(θ̂_pre; x_{1:s}) + ℓ(θ̂_post; x_{s+1:t}) − ℓ(θ̂_null; x_{1:t})

    and the penalty is:

        penalty(t) = v · (log t + √(log t))

    A candidate triggers an alarm when its score exceeds the calibrated
    threshold.  MLEs are computed by Newton's method; a warm-start one step
    from ``theta_init`` is used (or the sample-covariance heuristic when
    ``cov_parametrization=True``).

    Examples
    --------
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
        min_seg: int | None = None,
        cov_parametrization: bool = False,
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

        # --- Covariance parametrization validation ---
        if cov_parametrization:
            expected_v = n_features * (n_features + 1) // 2
            if v != expected_v or n_features <= 1:
                raise ValueError(
                    f"cov_parametrization=True requires v = p*(p+1)/2 with "
                    f"p = n_features > 1.  Got v={v}, n_features={n_features} "
                    f"(expected v={expected_v})."
                )

        # --- Build solver and GLR kernel ---
        if v == 1:
            if A_prime is None or A_dprime is None:
                raise ValueError(
                    "A_prime and A_dprime are required for scalar case (v=1)."
                )
            solver = make_newton_solver(A_prime, A_dprime)
            self._glr_kernel = _make_scalar_glr_kernel(
                A, solver, A_prime, A_dprime, theta_init, min_seg
            )
        else:
            if A_grad is None or A_hess is None:
                raise ValueError(
                    "A_grad and A_hess are required for vector case (v>1)."
                )
            solver = make_vector_newton_solver(A_grad, A_hess)
            self._glr_kernel = _make_vector_glr_kernel(
                A, solver, A_grad, A_hess, theta_init, min_seg, cov_parametrization
            )

        self._penalty_fn = _make_penalty(v)
        self._theta_init = theta_init

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
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_arr.size != self.n_features:
            raise ValueError(
                f"ExponentialFamilyGLR expected observation of size "
                f"{self.n_features}, got {x_arr.size}."
            )

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
            Penalised GLR score for each active candidate.  Scores are
            non-negative; candidates with too few observations on either
            side receive a score of 0.

        Raises
        ------
        ValueError
            If ``grid_states`` is empty.
        """
        if len(grid_states) == 0:
            raise ValueError("grid_states is empty.")

        before_stats = np.stack([gs.suff_stat for gs in grid_states])
        before_n = np.array([gs.n_samples for gs in grid_states], dtype=np.int64)

        raw_scores = self._glr_kernel(
            state.suff_stat, before_stats, state.n_samples, before_n
        )
        penalty = self._penalty_fn(state.n_samples)
        return raw_scores / penalty
