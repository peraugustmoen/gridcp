"""The online grid detector."""

from dataclasses import dataclass, field
from typing import Generic
import numpy as np
from numpy.typing import ArrayLike

from gridcp.new_api.typing import ScoreModel, TScoreState
from gridcp.new_api.utils import v2


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
    """State of the grid detector at a given time step."""

    running_score_state: TScoreState
    n_samples: int = 0
    candidate_score_states: list[TScoreState] = field(default_factory=list)
    grid: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class GridDetector:
    """The online grid detector.

    Owns:
    - A state.
    - A threshold.
    - The grid of candidate changepoints.

    """

    score: ScoreModel
    threshold: float = 1.0

    def __post_init__(self):
        """Validate the threshold and score object."""
        if self.threshold <= 0:
            raise ValueError("threshold must be positive.")
        if not isinstance(self.score, ScoreModel):
            raise TypeError("score must implement the ScoreModel protocol.")

    def init_state(self) -> DetectorState:
        """Return a fresh initial state with no observations seen."""
        return DetectorState(running_score_state=self.score.init_state())

    def update(
        self,
        state: DetectorState,
        x: ArrayLike,
    ) -> tuple[DetectorState, dict]:
        """Process a new observation and update the grid detector's state.

        Parameters
        ----------
        state : DetectorState
            Current state of the grid detector.
        x : ArrayLike
            New observation to process.

        Returns
        -------
        tuple[DetectorState, dict]
            Updated state and output dictionary.
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
        )

        if new_n_samples >= 2:
            penalised_scores = self.score.compute_penalised_scores(
                new_state.running_score_state, new_state.candidate_score_states
            )
            argmax = int(np.argmax(penalised_scores))
            max_score = float(penalised_scores[argmax])
            max_score_index = new_state.grid[argmax]
        else:
            max_score = 0.0
            max_score_index = 0

        alarm = max_score > self.threshold

        output = {
            "index": new_state.n_samples,
            "alarm": alarm,
            "max_score": max_score,
            "max_score_index": max_score_index,
        }
        return new_state, output
