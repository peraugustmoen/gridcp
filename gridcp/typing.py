"""Typing definitions for the grid detector API."""

from enum import Enum
from typing import Protocol, TypedDict, TypeVar, runtime_checkable
import numpy as np
from numpy.typing import ArrayLike

TScoreState = TypeVar("TScoreState")


class DetectorOutput(TypedDict):
    """Output dictionary returned by ``GridDetector.update()``.

    Attributes
    ----------
    n_samples : int
        Number of observations seen since the most recent reset.
        This is the detector's local time and is the value that returns to 0
        after a reset.
    alarm : bool
        Whether any score exceeded the threshold at this time step.
    max_score : float | np.ndarray
        Maximum penalised score across grid candidates.  Scalar for
        single-test scores, shape ``(K,)`` for multivariate scores.
    max_score_index : int | np.ndarray
        0-based index into the active candidate list (``state.grid``)
        that achieved the max score. Scalar for single-test scores,
        shape ``(K,)`` for multivariate scores.

        For ``n_samples < 2``, no candidate scores are available yet and this
        field is a placeholder (0 for scalar thresholds, zeros for vector
        thresholds).
    """

    n_samples: int
    alarm: bool
    max_score: float | np.ndarray
    max_score_index: int | np.ndarray


class PenaltyType(Enum):
    """Penalty mode for built-in score models.

    ``TIME_DEPENDENT``
        Time-growing penalty (e.g. log *t* + √log *t*) designed for false alarm
        probability control.
    ``CONSTANT``
        No time-dependent normalisation (penalty = 1). Scores are only centered,
        and the threshold absorbs all scaling.  Suitable for Average Run Length
        (ARL) control where the threshold is calibrated via Monte Carlo.
    """

    TIME_DEPENDENT = "time_dependent"
    CONSTANT = "constant"


@runtime_checkable
class ScoreModel(Protocol[TScoreState]):
    """Protocol for score computers used within the grid detector.

    A compliant class maintains per-candidate sufficient statistics and computes
    scores for all active candidates after each new observation.

    IMPORTANT: state objects returned by ``init_state``/``update`` are treated as
    immutable snapshots by ``GridDetector``. ``update`` must return a new state
    and must not mutate the input ``state`` in place.

    IMPORTANT: ``GridDetector`` calls ``compute_penalised_scores(state, grid_states)``
    and score implementations should derive any time-dependent penalty scaling
    from the provided state.  Accordingly, ``TScoreState`` must carry whatever
    time information the score needs (typically an ``n_samples`` counter updated
    by ``update``).  No separate ``n_samples_for_penalty`` argument is passed.

    The grid detector calls these methods; the implementation is free to choose
    any backend (NumPy, Numba, pandas, PyTorch, JAX, etc.).
    """

    @property
    def n_features(self) -> int:
        """Observation dimension expected by the score model."""
        ...

    def init_state(self) -> TScoreState:
        """Return fresh initial state with no observations seen."""
        ...

    def update(self, state: TScoreState, x: ArrayLike) -> TScoreState:
        """Incorporate a new observation into the global running statistic.

        Parameters
        ----------
        state : TScoreState
            Current state.
        x : ArrayLike
            New observation.

        Returns
        -------
        TScoreState
            Updated state.
        """
        ...

    def compute_penalised_scores(
        self,
        state: TScoreState,
        grid_states: list[TScoreState],
    ) -> np.ndarray:
        """Compute a penalised score for every active grid candidate.

        Parameters
        ----------
        state : TScoreState
            Global running state after the latest observation.  This is the
            authoritative source for any time-dependent penalty scaling; for
            example, built-in scores read ``state.n_samples`` from here.
            Custom ``TScoreState`` types must therefore include an ``n_samples``
            field (or equivalent) if time-dependent penalties are required.
        grid_states : list[TScoreState]
            Per-candidate state snapshots, one per active grid point.

        Returns
        -------
        np.ndarray, shape (len(grid_states),) or (len(grid_states), n_tests)
            Penalised score for each active candidate.  When the score model
            produces a single test statistic the shape is ``(G,)``.  When the
            model produces multiple test statistics (e.g. separate tests for
            mean vs variance), the shape is ``(G, K)`` where ``K`` is the
            number of tests.

        """
        ...
