"""Multivariate mean LR score under identity covariance."""

from dataclasses import dataclass, field

import numba as nb
import numpy as np

from gridcp.typing import ArrayLike
from gridcp.scores._score_helpers import as_obs


@nb.njit(cache=True)
def multivariate_mean_identity_cov_score(
    total_sum: np.ndarray,
    before_sums: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> np.ndarray:
    """Compute sparse and dense sq-CUSUM scores for all grid candidates.

    Parameters
    ----------
    total_sum : np.ndarray
        Sum of all observations, shape ``(n_features,)``.
    before_sums : np.ndarray
        Per-candidate sums, shape ``(G, n_features)``.
    total_samples : int
        Total number of observations seen so far.
    before_samples : np.ndarray
        Number of observations before each candidate, shape ``(G,)``.

    Returns
    -------
    np.ndarray
        Score matrix of shape ``(G, 2)``.  Column 0: sparse (max sq-CUSUM - 1).
        Column 1: dense (sum sq-CUSUM - p).
    """
    n_candidates = before_sums.shape[0]
    n_features = before_sums.shape[1]
    out = np.zeros((n_candidates, 2), dtype=np.float64)
    t = total_samples

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1

        if n1 == 0 or n2 == 0:
            out[i, 0] = 0.0
            out[i, 1] = 0.0
            continue

        before_weight = np.sqrt(n2 / (t * n1))
        after_weight = np.sqrt(n1 / (t * n2))

        max_val = 0.0
        sum_val = 0.0
        for j in range(n_features):
            s1_j = before_sums[i, j]
            s2_j = total_sum[j] - s1_j
            sq = (before_weight * s1_j - after_weight * s2_j) ** 2
            sum_val += sq
            if sq > max_val:
                max_val = sq

        out[i, 0] = max_val - 1.0
        out[i, 1] = sum_val - n_features

    return out


@dataclass(slots=True)
class MultivariateMeanIdentityCovState:
    """State for the MultivariateMeanIdentityCov score.

    Parameters
    ----------
    n_samples : int
        Number of samples seen so far.
    sum : np.ndarray
        Cumulative sum of observations, shape ``(n_features,)``.
    """

    n_samples: int = 0
    sum: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class MultivariateMeanIdentityCov:
    """Multivariate mean-change score under known identity covariance.

    Under no change, X_1, ..., X_t are i.i.d. N(μ, I_p); under a change at τ,
    X_1, ..., X_{τ-1} ~ N(μ₁, I_p) and X_τ, ..., X_t ~ N(μ₂, I_p) with μ₁ ≠ μ₂.

    **Score.**  For a candidate with n_1 pre-change and n_2 = t - n_1
    post-change observations, the per-coordinate squared CUSUM statistic is

        C_j = (n_1 n_2 / t) (x̄_{1,j} - x̄_{2,j})²,

    which under the null is chi-squared(1) for each j.

    **Aggregation.**  The p coordinate statistics are aggregated in two ways,
    both returned simultaneously as two score columns (``n_scores = 2``):

    - **Sparse** (column 0): ``max_j C_j - 1``, sensitive to changes
      confined to a small number of coordinates.  Centered by 1 (the
      chi-squared(1) mean).
    - **Dense** (column 1): ``sum_j C_j - p``, the full ``2 * LR`` under
      identity covariance.  Centered by p (the chi-squared(p) mean).

    **Centering and penalty.**  Both columns are centered as described above.
    When ``enable_penalty=True`` (default):

    - Sparse (column 0): divided by ``log(t) + log(p)``, from a
      non-asymptotic Gaussian concentration bound.
    - Dense (column 1): divided by ``sqrt(p log t) + log t``, from a
      non-asymptotic Gaussian concentration bound.

    When ``enable_penalty=False``, both columns are divided by 1.0 and the
    raw centered statistics are returned.

    **Sample size requirement.**  None.

    Parameters
    ----------
    n_features : int
        Dimension ``p`` of each observation vector.
    enable_penalty : bool, default=True
        If ``True``, divide each score column by its respective penalty.
        If ``False``, return raw centered scores with divisor 1.0.
    """

    n_features: int
    enable_penalty: bool = True

    @property
    def n_scores(self) -> int:
        """Number of scores returned by ``compute_penalized_scores``."""
        return 2

    def init_state(self) -> MultivariateMeanIdentityCovState:
        """Return a fresh initial state with no observations seen."""
        return MultivariateMeanIdentityCovState(
            sum=np.zeros(self.n_features, dtype=np.float64)
        )

    def update(
        self,
        state: MultivariateMeanIdentityCovState,
        x: ArrayLike,
    ) -> MultivariateMeanIdentityCovState:
        """Update the state with a new observation.

        Parameters
        ----------
        state : MultivariateMeanIdentityCovState
            Current state.
        x : ArrayLike
            New observation, shape ``(n_features,)``.

        Returns
        -------
        MultivariateMeanIdentityCovState
            Updated state.
        """
        x_arr = as_obs(x, self.n_features)
        return MultivariateMeanIdentityCovState(
            n_samples=state.n_samples + 1,
            sum=state.sum + x_arr,
        )

    def _compute_centered_scores(
        self,
        state: MultivariateMeanIdentityCovState,
        grid_states: list[MultivariateMeanIdentityCovState],
    ) -> np.ndarray:
        """Compute centered (but unpenalized) scores for every active grid candidate."""
        n = len(grid_states)
        before_sums = np.empty((n, self.n_features), dtype=np.float64)
        before_samples = np.empty(n, dtype=np.int64)
        for i, st in enumerate(grid_states):
            before_sums[i] = st.sum
            before_samples[i] = st.n_samples
        return multivariate_mean_identity_cov_score(
            state.sum,
            before_sums,
            state.n_samples,
            before_samples,
        )

    def _get_penalty(self, n_samples: int) -> np.ndarray:
        """Return the penalty divisor for the current sample size."""
        if self.enable_penalty:
            t = n_samples
            p = self.n_features
            return np.array(
                [
                    np.log(t) + np.log(p),
                    np.sqrt(p * np.log(t)) + np.log(t),
                ]
            )
        return np.ones(2)

    def compute_penalized_scores(
        self,
        state: MultivariateMeanIdentityCovState,
        grid_states: list[MultivariateMeanIdentityCovState],
    ) -> np.ndarray:
        """Compute penalized multivariate mean-change scores."""
        return self._compute_centered_scores(state, grid_states) / self._get_penalty(
            state.n_samples
        )
