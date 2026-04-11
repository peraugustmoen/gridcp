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
    global_n_samples : int
        Total number of observations seen across the full detector lifetime,
        including observations before resets when offset preservation is used.
        When a reset is performed with ``preserve_offset=False``, this counter
        also restarts from 0 because the detector intentionally discards the
        previous false-alarm time accounting.
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
    global_n_samples: int
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

    IMPORTANT: ``GridDetector`` now always calls
    ``compute_penalised_scores(..., n_samples_for_penalty=...)``. Custom score
    implementations must therefore accept this keyword argument. The detector
    does not silently fall back to an older two-argument signature, because
    doing so would make reset-related penalty bugs much harder to detect.

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
        n_samples_for_penalty: int | None = None,
    ) -> np.ndarray:
        """Compute a penalised score for every active grid candidate.

        Parameters
        ----------
        state : TScoreState
            Global running state after the latest observation.
        grid_states : list[TScoreState]
            Per-candidate state snapshots, one per active grid point.
        n_samples_for_penalty : int | None, default=None
            Optional sample count to use for time-dependent penalties. If
            ``None``, implementations should use ``state.n_samples``.
            ``GridDetector`` uses this argument to preserve long-run false-alarm
            semantics across resets: the score state itself is reset, but the
            penalty can continue using a cumulative sample count.

        Returns
        -------
        np.ndarray, shape (len(grid_states),) or (len(grid_states), n_tests)
            Penalised score for each active candidate.  When the score model
            produces a single test statistic the shape is ``(G,)``.  When the
            model produces multiple test statistics (e.g. separate tests for
            mean vs variance), the shape is ``(G, K)`` where ``K`` is the
            number of tests.

        Notes
        -----
        A reset-aware score implementation should usually compute its raw or
        centered statistic from the local ``state`` and ``grid_states``, but
        compute the penalty divisor from ``n_samples_for_penalty`` when it is
        provided. This keeps the sufficient statistics local to the post-reset
        segment while keeping the penalty on the original false-alarm time
        scale.
        """
        ...
