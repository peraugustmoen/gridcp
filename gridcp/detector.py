"""The online grid detector.

The detector maintains two notions of time:

- ``n_samples``: local sample count since the most recent reset.
- ``n_samples_offset``: cumulative count carried over from previous detector
    epochs when reset is performed with offset preservation enabled.

The sum ``n_samples_offset + n_samples`` is the detector's global sample count.
Built-in score models use this global count only for time-dependent penalties,
while the sufficient statistics themselves remain local to the current epoch.
This lets the detector forget pre-reset data while keeping false-alarm control
on the original time scale.
"""

from dataclasses import dataclass, field
from typing import Generic
import numpy as np
from numpy.typing import ArrayLike

from gridcp.typing import DetectorOutput, ScoreModel, TScoreState
from gridcp.utils import v2


def _update_grid(
    grid: list[int],
    candidate_score_states: list[TScoreState],
    prev_running_score_state: TScoreState,
) -> tuple[list[int], list[TScoreState]]:
    """Update the grid and per-candidate scores for the next time step.

    Update the grid from one sample to the next and the corresponding list of
    per-candidate score state snapshots. The grid holds candidate changepoint
    positions relative to the beginning of the time series.

    Parameters
    ----------
    grid : list[int]
        Current grid (candidate insertion times).
    candidate_score_states : list[TScoreState]
        Per-candidate score state snapshots, parallel to `grid`.
    prev_running_score_state : TScoreState
        Score state snapshot to append as the new candidate (the pre-update
        global state, captured before the current observation is processed).

    Returns
    -------
    grid : list[int]
        Updated grid.
    candidate_score_states : list[TScoreState]
        Updated per-candidate score state snapshots.
    """
    # list() makes a shallow copy, which is sufficient since both the grid and the
    # score states are immutable snapshots.
    new_grid = list(grid)
    new_candidate_score_states = list(candidate_score_states)

    # The last element of the grid is always n_samples - 1.
    prev_n_samples = new_grid[-1] + 1 if len(new_grid) > 0 else 0
    if prev_n_samples == 1 or prev_n_samples == 2:
        new_grid.pop(0)
        new_candidate_score_states.pop(0)
    elif prev_n_samples > 2:
        j = v2(prev_n_samples) + 1
        if j > 0:
            ind = 2 * j
            if ind < len(new_grid):
                removed = len(new_grid) - ind - 1
                new_grid.pop(removed)
                new_candidate_score_states.pop(removed)

    new_grid.append(prev_n_samples)
    new_candidate_score_states.append(prev_running_score_state)
    return new_grid, new_candidate_score_states


@dataclass
class DetectorState(Generic[TScoreState]):
    """State of the grid detector at a given time step.

    Parameters
    ----------
    running_score_state : TScoreState
        Current running score state for the active post-reset epoch.
    n_samples : int, default=0
        Number of observations processed since the most recent reset.
    n_samples_offset : int, default=0
        Number of observations processed before the current epoch and preserved
        for global penalty-time accounting.
    candidate_score_states : list[TScoreState]
        Score-state snapshots for the active logarithmic grid.
    grid : list[int]
        Candidate changepoint locations for the active epoch, expressed in the
        local time scale ``[0, n_samples)``.
    """

    running_score_state: TScoreState
    n_samples: int = 0
    n_samples_offset: int = 0
    candidate_score_states: list[TScoreState] = field(default_factory=list)
    grid: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class GridDetector:
    """The online grid detector.

    The detector owns the active score model, the alarm threshold, and the
    logarithmic grid of candidate changepoints.

    Reset semantics
    ---------------
    Resetting is handled by the module-level function
    ``reset_detector_state(state, detector, preserve_offset=...)``. Keeping
    reset external preserves the detector's functional update style and avoids
    hidden state transitions in ``update()``.

    Threshold semantics
    -------------------
    For scalar score outputs, threshold must be scalar.

    For multivariate score outputs with shape ``(G, K)``:
    - If threshold is scalar, it is broadcast to shape ``(K,)`` on each call.
    - If threshold is a vector, it must have shape ``(K,)``.
    """

    score: ScoreModel
    threshold: float | np.ndarray = 1.0

    def _resolve_threshold_for_scores(
        self,
        threshold: np.ndarray,
        penalised_scores: np.ndarray,
    ) -> np.ndarray:
        """Return threshold shaped for score comparison.

        For multivariate score outputs, a scalar threshold is broadcast
        to all score components.
        """
        if penalised_scores.ndim == 1:
            if threshold.ndim != 0:
                raise ValueError(
                    "threshold shape mismatch: per-candidate penalised score is "
                    "scalar (K=1), so threshold must be scalar; "
                    f"got threshold shape {threshold.shape}."
                )
            return threshold

        if penalised_scores.ndim == 2:
            n_tests = penalised_scores.shape[1]
            if threshold.ndim == 0:
                return np.full(n_tests, float(threshold), dtype=np.float64)

            if threshold.ndim != 1 or threshold.shape[0] != n_tests:
                raise ValueError(
                    "threshold shape mismatch: per-candidate penalised score "
                    f"dimension is K={n_tests}, so threshold must have shape "
                    f"({n_tests},); got threshold shape {threshold.shape}."
                )
            return threshold

        raise ValueError(
            "penalised_scores must be a 1-D or 2-D array, "
            f"got shape {penalised_scores.shape}."
        )

    def __post_init__(self):
        """Validate the threshold and score object."""
        th = np.asarray(self.threshold, dtype=np.float64)
        if th.ndim == 0:
            if float(th) <= 0:
                raise ValueError("threshold must be positive.")
        elif th.ndim == 1:
            if np.any(th <= 0):
                raise ValueError("All threshold entries must be positive.")
        else:
            raise ValueError("threshold must be a scalar or 1-D array.")
        if not isinstance(self.score, ScoreModel):
            raise TypeError("score must implement the ScoreModel protocol.")

    def init_state(self) -> DetectorState:
        """Return a fresh initial state with no observations seen.

        The returned state has empty grid history, ``n_samples = 0``, and
        ``n_samples_offset = 0``.
        """
        return DetectorState(running_score_state=self.score.init_state())

    def update(
        self,
        state: DetectorState,
        x: ArrayLike,
    ) -> tuple[DetectorState, DetectorOutput]:
        """Process a new observation and update the grid detector's state.

        Parameters
        ----------
        state : DetectorState
            Current state of the grid detector.
        x : ArrayLike
            New observation to process.

        Returns
        -------
        tuple[DetectorState, DetectorOutput]
            Updated state and output dictionary.

            The output always includes both local and global time:

            - ``output["n_samples"]``: local post-reset sample count
            - ``output["global_n_samples"]``: cumulative count used for
              penalty-time accounting when offset preservation is active
        """
        new_n_samples = state.n_samples + 1
        new_running_score_state = self.score.update(state.running_score_state, x)
        new_grid, new_candidate_score_states = _update_grid(
            state.grid, state.candidate_score_states, state.running_score_state
        )
        new_state = DetectorState(
            running_score_state=new_running_score_state,
            candidate_score_states=new_candidate_score_states,
            grid=new_grid,
            n_samples=new_n_samples,
            n_samples_offset=state.n_samples_offset,
        )

        global_n_samples = new_state.n_samples_offset + new_state.n_samples

        if new_n_samples >= 2:
            penalised_scores = self.score.compute_penalised_scores(
                new_state.running_score_state,
                new_state.candidate_score_states,
                n_samples_for_penalty=global_n_samples,
            )
            th = np.asarray(self.threshold, dtype=np.float64)
            comparison_threshold = self._resolve_threshold_for_scores(
                th, penalised_scores
            )

            if penalised_scores.ndim == 1:
                # Scalar test: shape (G,)
                argmax = int(np.argmax(penalised_scores))
                max_score = float(penalised_scores[argmax])
                max_score_index = argmax
            else:
                # Multivariate tests: shape (G, K)
                # Per-test max over grid candidates.
                max_score = np.max(penalised_scores, axis=0)
                argmax_per_test = np.argmax(penalised_scores, axis=0)
                max_score_index = argmax_per_test.astype(np.int64, copy=False)
        else:
            th = np.asarray(self.threshold, dtype=np.float64)
            comparison_threshold = th
            if th.ndim == 0:
                max_score = 0.0
                max_score_index = 0
            else:
                n_tests = th.shape[0]
                max_score = np.zeros(n_tests, dtype=np.float64)
                max_score_index = np.zeros(n_tests, dtype=np.int64)

        alarm = bool(np.any(np.asarray(max_score) > comparison_threshold))

        output_global_n_samples = new_state.n_samples_offset + new_state.n_samples

        output: DetectorOutput = {
            "n_samples": new_state.n_samples,
            "global_n_samples": output_global_n_samples,
            "alarm": alarm,
            "max_score": max_score,
            "max_score_index": max_score_index,
        }
        return new_state, output


def reset_detector_state(
    state: DetectorState,
    detector: GridDetector,
    *,
    preserve_offset: bool = True,
) -> DetectorState:
    """Reset detector history while optionally preserving global sample time.

    Resetting discards all previously stored sufficient statistics and grid
    candidates, so the detector behaves as though it has just started on a
    new segment. The only quantity that may survive the reset is the sample
    count offset used for time-dependent penalty scaling.

    Parameters
    ----------
    state : DetectorState
        Current detector state.
    detector : GridDetector
        Detector instance used to initialize a fresh score state.
    preserve_offset : bool, default=True
        If True, preserve cumulative sample count for penalty scaling by
        advancing ``n_samples_offset`` by the current local ``n_samples``.
        If False, fully reset cumulative sample tracking.

    Returns
    -------
    DetectorState
        Fresh detector state with cleared grid/candidate history and a new
        score state from ``detector.score.init_state()``.

    Notes
    -----
    ``preserve_offset=True`` is the statistically conservative choice when
    the threshold is interpreted through a long-run false-alarm guarantee.
    ``preserve_offset=False`` intentionally restarts that time scale.
    """
    next_offset = 0
    if preserve_offset:
        next_offset = state.n_samples_offset + state.n_samples

    return DetectorState(
        running_score_state=detector.score.init_state(),
        n_samples=0,
        n_samples_offset=next_offset,
        candidate_score_states=[],
        grid=[],
    )
