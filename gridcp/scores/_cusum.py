"""The CUSUM score for a change in mean with known unit variance."""

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

# Per-coordinate degrees of freedom of the squared CUSUM statistic.
_CUSUM_DF = 1


@nb.njit(cache=True)
def cusum_score(
    total_sum: np.ndarray,
    before_sums: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> np.ndarray:
    """Per-coordinate squared CUSUM statistic for all grid candidates.

    Parameters
    ----------
    total_sum : np.ndarray
        Cumulative sum over all observations, shape ``(n_features,)``.
    before_sums : np.ndarray
        Per-candidate sums of observations, shape ``(G, n_features)``.
    total_samples : int
        Total number of observations seen so far.
    before_samples : np.ndarray
        First post-change index (0-based) for each candidate, shape ``(G,)``.
        Equals the pre-change sample count: ``data[0:n1]`` is pre-change.

    Returns
    -------
    np.ndarray
        Raw (uncentered, unaggregated) per-coordinate squared CUSUM, shape
        ``(G, n_features)``.  Each entry is
        ``C_j = (n_1 n_2 / t) (x̄_{1,j} - x̄_{2,j})²``.  Candidates with
        ``n1 == 0`` or ``n2 == 0`` (where the CUSUM is undefined) are filled
        with zeros and flagged invalid by the caller.
    """
    n_candidates = before_sums.shape[0]
    n_features = before_sums.shape[1]
    out = np.zeros((n_candidates, n_features), dtype=np.float64)
    t = total_samples

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1
        if n1 == 0 or n2 == 0:
            continue

        before_weight = np.sqrt(n2 / (t * n1))
        after_weight = np.sqrt(n1 / (t * n2))
        for j in range(n_features):
            s1 = before_sums[i, j]
            s2 = total_sum[j] - s1
            out[i, j] = (before_weight * s1 - after_weight * s2) ** 2

    return out


@dataclass(slots=True)
class CUSUMState:
    """State for the :class:`CUSUM` score.

    Parameters
    ----------
    n_samples : int
        Number of samples seen so far.
    sum : np.ndarray
        Cumulative sum of the samples seen so far, shape ``(n_features,)``.
    """

    n_samples: int = 0
    sum: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class CUSUM:
    """CUSUM score for a change in the mean with known unit variance.

    Under no change, X_1, ..., X_t are i.i.d. N(μ, I_p); under a change at τ,
    X_1, ..., X_{τ-1} ~ N(μ₁, I_p) and X_τ, ..., X_t ~ N(μ₂, I_p) with μ₁ ≠ μ₂.

    **Score.**  For a candidate with n_1 pre-change and n_2 = t - n_1
    post-change observations, the per-coordinate squared CUSUM statistic is

        C_j = (n_1 n_2 / t) (x̄_{1,j} - x̄_{2,j})²,

    the exact ``2 * LR`` under known unit variance.  Under the null, C_j is
    chi-squared(1) for each coordinate j.

    **Aggregation.**  The p coordinate statistics are combined according to the
    ``aggregation`` keyword (see Parameters).  ``"max"`` is sensitive to changes
    concentrated in any single coordinate; ``"sum"`` is the dense
    ``2 * LR = sum_j C_j``; ``"max-sum"`` returns both as two columns;
    ``None`` returns all p per-coordinate statistics.

    **Centering and penalty.**  Each output column is centered by subtracting
    its degrees of freedom ``df`` (the chi-squared mean) inside the score.  When
    ``enable_penalty=True`` (default), the centered column is divided by
    ``chi2_max_bound(M, df, t)``; when ``enable_penalty=False`` the divisor is
    1.0 and the centered statistic is returned.

    **Sample size requirement.**  Candidates with ``n1 == 0`` or ``n2 == 0``
    return 0 (the CUSUM is undefined there).

    Parameters
    ----------
    n_features : int, default=1
        Number of features ``p`` in the input data.
    aggregation : {"max", "sum", "max-sum", None, "None"}, default="max"
        How to combine the p per-coordinate statistics across the feature axis.
        ``"max"`` (1 column), ``"sum"`` (1 column), ``"max-sum"`` (2 columns,
        max then sum), ``None``/``"None"`` (p columns, one per feature).
    enable_penalty : bool, default=True
        If ``True``, divide each centered column by ``chi2_max_bound(M, df, t)``.
        If ``False``, return the raw centered statistic with divisor 1.0.

    Examples
    --------
    >>> from gridcp.scores import CUSUM
    >>> score = CUSUM(n_features=2)
    >>> state = score.init_state()
    >>> state = score.update(state, [1.0, 2.0])
    >>> state = score.update(state, [1.5, 2.5])
    >>> state
    CUSUMState(n_samples=2, sum=array([2.5, 4.5]))
    """

    n_features: int = 1
    aggregation: object = "max"
    enable_penalty: bool = True

    def __post_init__(self):
        """Validate the ``aggregation`` keyword at construction time."""
        aggregation_mode_code(self.aggregation)

    @property
    def n_scores(self) -> int:
        """Number of scores returned by ``compute_penalized_scores``."""
        return len(aggregation_dims(self.aggregation, self.n_features, _CUSUM_DF))

    def init_state(self) -> CUSUMState:
        """Return a fresh initial state with no observations seen."""
        return CUSUMState(sum=np.zeros(self.n_features))

    def update(self, state: CUSUMState, x: ArrayLike) -> CUSUMState:
        """Update the state with a new observation.

        Parameters
        ----------
        state : CUSUMState
            Current state.
        x : ArrayLike
            New observation, shape ``(n_features,)``.

        Returns
        -------
        CUSUMState
            Updated state.
        """
        x_arr = as_obs(x, self.n_features)
        return CUSUMState(n_samples=state.n_samples + 1, sum=state.sum + x_arr)

    def compute_penalized_scores(
        self,
        state: CUSUMState,
        grid_states: list[CUSUMState],
    ) -> np.ndarray:
        """Compute a penalized score for every active grid candidate."""
        if len(grid_states) == 0:
            raise ValueError("grid_states is empty.")

        grid_sums = np.stack([st.sum for st in grid_states])
        before_samples = np.array([st.n_samples for st in grid_states], dtype=np.int64)
        t = state.n_samples

        raw = cusum_score(
            total_sum=state.sum,
            before_sums=grid_sums,
            total_samples=t,
            before_samples=before_samples,
        )
        cols = aggregate_features(raw, aggregation_mode_code(self.aggregation))
        dims = aggregation_dims(self.aggregation, self.n_features, _CUSUM_DF)

        out = np.empty((cols.shape[0], len(dims)), dtype=np.float64)
        for j, (n_terms, df) in enumerate(dims):
            penalty = chi2_max_bound(n_terms, df, t) if self.enable_penalty else 1.0
            out[:, j] = (cols[:, j] - df) / penalty

        # Candidates with no valid split (n1 == 0 or n2 == 0) score 0.
        invalid = (before_samples == 0) | (before_samples == t)
        out[invalid, :] = 0.0
        return out
