"""The online grid detector."""

from dataclasses import dataclass, field
from typing import Generic
import numpy as np
from numpy.typing import ArrayLike

from gridcp.new_api.typing import GridScore, ScoreState
from gridcp.new_api.utils import v2


def _update_grid(
    grid: list[int], grid_score_states: list[ScoreState], prev_score_state: ScoreState
) -> tuple[list[int], list[ScoreState]]:
    """Update the grid and per-candidate scores for the next time step.

    Update the grid from one sample to the next and the
    corresponding list of per-candidate score state snapshots. The grid in this case
    holds the candidate changepoint positions relative to the beginning of the the
    time series, not relative to the current time step.

    Parameters
    ----------
    old_grid : list[int]
        Current grid (candidate insertion times).
    grid_score_states : list[ScoreState]
        Per-candidate score state snapshots, parallel to `grid`.
    score_state : ScoreState
        Score state snapshot to append as the new candidate (the pre-update
        global state, captured before the current observation is processed).
    n_samples : int
        Current time step (number of observations seen so far).
    copy : bool, optional
        If True (default), operate on copies of `old_grid` and
        `old_score_states`, leaving the originals unchanged.
        If False, mutate both lists in-place.

    Returns
    -------
    n_samples : int
        Updated time step (should be `n_samples + 1`).
    grid : list[int]
        Updated grid.
    grid_score_states : list
        Updated per-candidate score state snapshots.
    """
    # list() makes a shallow copy, which is sufficient since both the grid and the
    # score states are immutable snapshots.
    new_grid = list(grid)
    new_grid_score_states = list(grid_score_states)

    # The last element of the grid is always n_samples - 1.
    prev_n_samples = new_grid[-1] + 1 if len(new_grid) > 0 else 0
    if prev_n_samples == 1 or prev_n_samples == 2:
        new_grid.pop(0)
        new_grid_score_states.pop(0)
    elif prev_n_samples > 2:
        j = v2(prev_n_samples) + 1
        if j > 0:
            ind = 2 * j
            if ind < len(new_grid):
                removed = len(new_grid) - ind - 1
                new_grid.pop(removed)
                new_grid_score_states.pop(removed)

    new_grid.append(prev_n_samples)
    new_grid_score_states.append(prev_score_state)
    return new_grid, new_grid_score_states


@dataclass
class GridDetectorState(Generic[ScoreState]):
    """State of the grid detector at a given time step."""

    score_state: ScoreState
    n_samples: int = 0
    grid_score_states: list[ScoreState] = field(default_factory=list)
    grid: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class GridDetector:
    """The online grid detector.

    Owns:
    - A state.
    - A penalty parameter.
    - The grid of candidate changepoints.

    """

    score: GridScore
    penalty_factor: float = 1.0

    def __post_init__(self):
        """Validate the penalty factor and score object."""
        if self.penalty_factor <= 0:
            raise ValueError("penalty_factor must be positive.")
        if not isinstance(self.score, GridScore):
            raise TypeError("score must implement the GridScore protocol.")

    def init_state(self) -> GridDetectorState:
        """Return a fresh initial state with no observations seen."""
        return GridDetectorState(score_state=self.score.init_state())

    def update(
        self,
        state: GridDetectorState,
        x: ArrayLike,
    ) -> tuple[GridDetectorState, dict]:
        """Process a new observation and update the grid detector's state.
        
        Parameters
        ----------
        state : GridDetectorState
            Current state of the grid detector.
        x : ArrayLike
            New observation to process.

        Returns
        -------
        tuple[GridDetectorState, dict]
            Updated state and output dictionary.
        """
        new_n_samples = state.n_samples + 1
        new_score_state = self.score.update(state.score_state, x)
        new_grid, new_grid_score_states = _update_grid(
            state.grid, state.grid_score_states, state.score_state
        )
        new_state = GridDetectorState(
            score_state=new_score_state,
            grid_score_states=new_grid_score_states,
            grid=new_grid,
            n_samples=new_n_samples,
        )

        if new_n_samples >= 2:
            penalised_scores = self.score.compute_penalised_scores(
                new_state.score_state, new_state.grid_score_states
            )
            argmax = int(np.argmax(penalised_scores))
            max_score = float(penalised_scores[argmax])
            max_score_index = new_state.grid[argmax]
        else:
            max_score = 0.0
            max_score_index = 0

        alarm = max_score > self.penalty_factor

        output = {
            "index": new_state.n_samples,
            "alarm": alarm,
            "max_score": max_score,
            "max_score_index": max_score_index,
        }
        return new_state, output
