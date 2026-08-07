"""Mean-change score with an unknown (diagonal or full) covariance (Gaussian LR)."""

import warnings
from dataclasses import dataclass, field

import numba as nb
import numpy as np

from gridcp.scores._aggregation import (
    aggregate_features,
    aggregation_dims,
    aggregation_mode_code,
    chi2_max_bound,
)
from gridcp.scores._score_helpers import as_obs
from gridcp.typing import ArrayLike

# Per-coordinate degrees of freedom of the diagonal mean-change LR statistic.
_GAUSSIAN_MEAN_DF = 1

# Allowed values for the ``cov_estimate`` argument.
_COV_ESTIMATES = ("diagonal", "full")


@nb.njit(cache=True)
def gaussian_mean_score(
    total_stats: np.ndarray,
    before_stats: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean-change LR (diagonal, unknown variance) for all candidates.

    Parameters
    ----------
    total_stats : np.ndarray, shape (2, n_features)
        Running sufficient statistics ``[sum(x), sum(x^2)]``.
    before_stats : np.ndarray, shape (n_candidates, 2, n_features)
        Candidate-wise sufficient statistics.
    total_samples : int
        Total number of samples seen.
    before_samples : np.ndarray, shape (n_candidates,)
        First post-change index (0-based) for each candidate.

    Returns
    -------
    raw : np.ndarray, shape (n_candidates, n_features)
        Raw (uncentered, unaggregated) per-feature LR. Features with a
        degenerate variance estimate are set to 0.
    valid : np.ndarray, shape (n_candidates,)
        Boolean flag; ``False`` for candidates with ``n1 <= 2``/``n2 <= 2`` or
        no valid feature (these score 0 after centering).
    """
    n_candidates = before_samples.shape[0]
    n_features = total_stats.shape[1]
    raw = np.zeros((n_candidates, n_features), dtype=np.float64)
    valid = np.zeros(n_candidates, dtype=np.bool_)

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = total_samples - n1
        if n1 <= 2 or n2 <= 2:
            continue

        has_valid_feature = False
        for k in range(n_features):
            total_sum = total_stats[0, k]
            total_sum2 = total_stats[1, k]
            sum1 = before_stats[i, 0, k]
            sum2_1 = before_stats[i, 1, k]

            sum2 = total_sum - sum1
            sum2_2 = total_sum2 - sum2_1

            sigma_null = (
                total_sum2 - total_sum * total_sum / total_samples
            ) / total_samples
            sigma_alt = (
                (sum2_1 - sum1 * sum1 / n1) + (sum2_2 - sum2 * sum2 / n2)
            ) / total_samples

            if sigma_null <= 0.0 or sigma_alt <= 0.0:
                continue

            has_valid_feature = True
            raw[i, k] = total_samples * np.log(sigma_null) - total_samples * np.log(
                sigma_alt
            )

        valid[i] = has_valid_feature

    return raw, valid


@nb.njit(cache=True)
def gaussian_mean_full_covariance_score(
    total_sum: np.ndarray,
    total_sum_outer: np.ndarray,
    before_sums: np.ndarray,
    before_sum_outers: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Compute pooled-within-group (full-covariance) LR scores for all candidates.

    Parameters
    ----------
    total_sum : np.ndarray
        Sum of all observations, shape ``(n_features,)``.
    total_sum_outer : np.ndarray
        Outer-product sum over all observations, shape ``(n_features, n_features)``.
    before_sums : np.ndarray
        Per-candidate sums, shape ``(G, n_features)``.
    before_sum_outers : np.ndarray
        Per-candidate outer-product sums, shape ``(G, n_features, n_features)``.
    total_samples : int
        Total number of observations seen so far.
    before_samples : np.ndarray
        First post-change index (0-based) for each candidate, shape ``(G,)``.
    n_features : int
        Dimension ``p`` of each observation vector.

    Returns
    -------
    np.ndarray
        Centered LR scores for each candidate, shape ``(G,)``. Each value is
        ``2 * LR - p`` where ``2 * LR = t * (logdet Σ̂_null - logdet Σ̂_alt)``.
        Returns the zero vector when ``t < 2*p + 2`` or when the null covariance
        matrix is singular; a candidate is 0 when its alternative covariance is
        singular.
    """
    n_candidates = before_sums.shape[0]
    out = np.zeros(n_candidates, dtype=np.float64)
    t = total_samples
    p = n_features
    df = float(p)

    if t < 2 * p + 2:
        return out

    s_tot = total_sum
    sxx_tot = total_sum_outer

    sigma_null = (sxx_tot - np.outer(s_tot, s_tot) / t) / t
    sign_null, logdet_null = np.linalg.slogdet(sigma_null)
    if sign_null <= 0:
        return out

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1

        s1 = before_sums[i]
        sxx1 = before_sum_outers[i]
        s2 = s_tot - s1
        sxx2 = sxx_tot - sxx1

        sigma_alt = (
            (sxx1 - np.outer(s1, s1) / n1) + (sxx2 - np.outer(s2, s2) / n2)
        ) / t
        sign1, logdet1 = np.linalg.slogdet(sigma_alt)
        if sign1 <= 0:
            out[i] = 0.0
            continue

        two_lr = t * (logdet_null - logdet1)
        out[i] = two_lr - df

    return out


@dataclass(slots=True)
class GaussianMeanState:
    """Running sufficient statistics for the :class:`GaussianMean` score.

    The populated fields depend on the score's ``cov_estimate``:

    Parameters
    ----------
    n_samples : int
        Number of samples seen so far.
    stats : np.ndarray
        Diagonal mode only: ``[sum(x), sum(x²)]``, shape ``(2, n_features)``
        (O(p)).
    sum : np.ndarray
        Full mode only: cumulative sum ``Σx``, shape ``(n_features,)``.
    sum_outer : np.ndarray
        Full mode only: cumulative outer-product sum ``Σxxᵀ``, shape
        ``(n_features, n_features)``.
    """

    n_samples: int = 0
    stats: np.ndarray = field(
        default_factory=lambda: np.empty((2, 0), dtype=np.float64)
    )
    sum: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    sum_outer: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )


@dataclass(frozen=True, slots=True)
class GaussianMean:
    """Mean-change score with an unknown common covariance.

    Under no change, X_1, ..., X_t are i.i.d. N(μ, Σ) with Σ unknown; under a
    change at τ the mean shifts from μ₁ to μ₂ with a common unknown covariance Σ.
    The nuisance covariance Σ is estimated according to ``cov_estimate``.

    **Score.**

    - ``cov_estimate="diagonal"`` (default): Σ is taken diagonal, with a separate
      unknown variance per feature. The per-feature twice profile
      log-likelihood ratio ``2 * LR = t * (log σ̂²_null - log σ̂²_alt)`` is
      computed for each feature (asymptotically chi-squared(1) under the null,
      Wilks) and combined across features by ``aggregation``.
    - ``cov_estimate="full"``: Σ is a full unknown covariance, pooled within
      group. The joint statistic
      ``2 * LR = t * (logdet Σ̂_null - logdet Σ̂_alt)`` is asymptotically
      chi-squared(p) under the null and forms a single test statistic.

    **Aggregation.** Applies only to ``cov_estimate="diagonal"``: the p
    per-feature statistics are combined according to the ``aggregation`` keyword
    (``"max"`` by default; see Parameters). For ``cov_estimate="full"`` the
    score is a single joint statistic (``n_scores = 1``) and ``aggregation`` has
    no effect, so passing a non-default value emits a ``UserWarning``.

    **Centering and penalty.** Each output column is centered by subtracting its
    degrees of freedom ``df`` inside the score (``df = 1`` per feature for
    diagonal; ``df = p`` for full). When ``enable_penalty=True`` (default) the
    centered column is divided by ``chi2_max_bound(M, df, t)``
    (``chi2_max_bound(1, p, t)`` for full); when ``enable_penalty=False`` the
    divisor is 1.0.

    **Sample size requirement.**

    - diagonal: candidates with n_1 ≤ 2 or n_2 ≤ 2 return 0.0 (degenerate
      variance estimates); features with a non-positive variance estimate are
      skipped.
    - full: the score returns 0 whenever ``t < 2*p + 2`` or the pooled (null)
      covariance is singular; an individual candidate is 0 when its alternative
      covariance is singular.

    Parameters
    ----------
    n_features : int, default=1
        Number of features ``p`` expected in each observation.
    cov_estimate : {"diagonal", "full"}, default="diagonal"
        Nuisance covariance model. ``"diagonal"`` estimates a per-feature
        variance (O(p) running state, supports ``aggregation``); ``"full"``
        estimates a full p×p covariance (O(p²) running state, single joint
        statistic).
    aggregation : {"max", "sum", "max-sum", None, "None"}, default="max"
        How to combine the p per-feature statistics across the feature axis when
        ``cov_estimate="diagonal"``. ``"max"`` (1 column), ``"sum"`` (1 column),
        ``"max-sum"`` (2 columns, max then sum), ``None``/``"None"`` (p columns,
        one per feature). Ignored (with a ``UserWarning`` if non-default) when
        ``cov_estimate="full"``.
    enable_penalty : bool, default=True
        If ``True``, divide each centered column by ``chi2_max_bound(M, df, t)``;
        if ``False``, return the raw centered statistic with divisor 1.0.
    """

    n_features: int = 1
    cov_estimate: str = "diagonal"
    aggregation: object = "max"
    enable_penalty: bool = True

    def __post_init__(self):
        """Validate inputs and warn on an ignored ``aggregation``."""
        if self.n_features < 1:
            raise ValueError("n_features must be >= 1.")
        if self.cov_estimate not in _COV_ESTIMATES:
            raise ValueError(
                f"cov_estimate must be one of {_COV_ESTIMATES}, got "
                f"{self.cov_estimate!r}."
            )
        aggregation_mode_code(self.aggregation)  # validate the aggregation value
        if self.cov_estimate == "full" and self.aggregation != "max":
            warnings.warn(
                "aggregation is ignored when cov_estimate='full' (the "
                "full-covariance LR produces a single joint statistic).",
                UserWarning,
                stacklevel=2,
            )

    @property
    def n_scores(self) -> int:
        """Number of scores returned by ``compute_penalized_scores``."""
        if self.cov_estimate == "full":
            return 1
        return len(
            aggregation_dims(self.aggregation, self.n_features, _GAUSSIAN_MEAN_DF)
        )

    def init_state(self) -> GaussianMeanState:
        """Return a fresh initial state with no observations seen."""
        if self.cov_estimate == "full":
            return GaussianMeanState(
                sum=np.zeros(self.n_features, dtype=np.float64),
                sum_outer=np.zeros(
                    (self.n_features, self.n_features), dtype=np.float64
                ),
            )
        return GaussianMeanState(stats=np.zeros((2, self.n_features), dtype=np.float64))

    def update(self, state: GaussianMeanState, x: ArrayLike) -> GaussianMeanState:
        """Update the state with a new observation.

        Parameters
        ----------
        state : GaussianMeanState
            Current state.
        x : ArrayLike
            New observation, shape ``(n_features,)``.

        Returns
        -------
        GaussianMeanState
            Updated state.
        """
        x_arr = as_obs(x, self.n_features)
        if self.cov_estimate == "full":
            return GaussianMeanState(
                n_samples=state.n_samples + 1,
                sum=state.sum + x_arr,
                sum_outer=state.sum_outer + np.outer(x_arr, x_arr),
            )
        increment = np.vstack((x_arr, x_arr * x_arr))
        return GaussianMeanState(
            n_samples=state.n_samples + 1,
            stats=state.stats + increment,
        )

    def compute_penalized_scores(
        self,
        state: GaussianMeanState,
        grid_states: list[GaussianMeanState],
    ) -> np.ndarray:
        """Compute penalized LR score at every active grid candidate."""
        if len(grid_states) == 0:
            raise ValueError("grid_states is empty.")
        if self.cov_estimate == "full":
            return self._compute_full(state, grid_states)
        return self._compute_diagonal(state, grid_states)

    def _compute_diagonal(
        self,
        state: GaussianMeanState,
        grid_states: list[GaussianMeanState],
    ) -> np.ndarray:
        """Per-feature diagonal path (O(p) state, supports ``aggregation``)."""
        total_stats = state.stats
        grid_stats = np.stack([st.stats for st in grid_states])
        before_samples = np.array([st.n_samples for st in grid_states], dtype=np.int64)
        t = state.n_samples

        raw, valid = gaussian_mean_score(
            total_stats=total_stats,
            before_stats=grid_stats,
            total_samples=t,
            before_samples=before_samples,
        )
        cols = aggregate_features(raw, aggregation_mode_code(self.aggregation))
        dims = aggregation_dims(self.aggregation, self.n_features, _GAUSSIAN_MEAN_DF)

        out = np.empty((cols.shape[0], len(dims)), dtype=np.float64)
        for j, (n_terms, df) in enumerate(dims):
            penalty = chi2_max_bound(n_terms, df, t) if self.enable_penalty else 1.0
            out[:, j] = (cols[:, j] - df) / penalty

        out[~valid, :] = 0.0
        return out

    def _compute_full(
        self,
        state: GaussianMeanState,
        grid_states: list[GaussianMeanState],
    ) -> np.ndarray:
        """Joint full-covariance path (single statistic, O(p²) state)."""
        n = len(grid_states)
        p = self.n_features
        before_sums = np.empty((n, p), dtype=np.float64)
        before_sum_outers = np.empty((n, p, p), dtype=np.float64)
        before_samples = np.empty(n, dtype=np.int64)
        for i, st in enumerate(grid_states):
            before_sums[i] = st.sum
            before_sum_outers[i] = st.sum_outer
            before_samples[i] = st.n_samples
        t = state.n_samples

        scores = gaussian_mean_full_covariance_score(
            state.sum,
            state.sum_outer,
            before_sums,
            before_sum_outers,
            t,
            before_samples,
            p,
        )
        penalty = chi2_max_bound(1, p, t) if self.enable_penalty else 1.0
        return (scores / penalty)[:, np.newaxis]
