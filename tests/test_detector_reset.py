from dataclasses import dataclass, field

import numpy as np
import pytest

from gridcp.detector import GridDetector
from gridcp.scores import MeanCUSUM


@dataclass(slots=True)
class _SpyState:
    n_samples: int = 0
    total: float = 0.0


@dataclass(slots=True)
class _SpyScore:
    n_features: int = 1
    last_penalty_n_samples: int | None = field(default=None, init=False)

    def init_state(self) -> _SpyState:
        return _SpyState()

    def update(self, state: _SpyState, x) -> _SpyState:
        val = float(np.asarray(x, dtype=np.float64).reshape(-1)[0])
        return _SpyState(n_samples=state.n_samples + 1, total=state.total + val)

    def compute_penalised_scores(
        self,
        state: _SpyState,
        grid_states: list[_SpyState],
        n_samples_for_penalty: int | None = None,
    ) -> np.ndarray:
        penalty_n = (
            state.n_samples if n_samples_for_penalty is None else n_samples_for_penalty
        )
        self.last_penalty_n_samples = penalty_n
        return np.full(len(grid_states), float(penalty_n), dtype=np.float64)


def _piecewise_constant_stream(
    changepoint: int = 50,
    total_n: int = 100,
    pre_mean: float = 0.0,
    post_mean: float = 8.0,
) -> list[np.ndarray]:
    values = np.concatenate(
        [
            np.full(changepoint, pre_mean, dtype=np.float64),
            np.full(total_n - changepoint, post_mean, dtype=np.float64),
        ]
    )
    return [np.array([value], dtype=np.float64) for value in values]


def _run_stream(detector: GridDetector, stream: list[np.ndarray]):
    state = detector.init_state()
    outputs = []
    for x in stream:
        state, out = detector.update(state, x)
        outputs.append(out)
    return state, outputs


def _run_until_first_alarm(detector: GridDetector, stream: list[np.ndarray]):
    state = detector.init_state()
    outputs = []
    alarm_index = None
    for i, x in enumerate(stream):
        state, out = detector.update(state, x)
        outputs.append(out)
        if out["alarm"]:
            alarm_index = i
            break
    return state, outputs, alarm_index


def test_output_includes_local_and_global_n_samples():
    """Check that detector outputs report both local and global sample counts."""
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=1e6)
    state = detector.init_state()

    state, out1 = detector.update(state, np.array([0.0]))
    assert out1["n_samples"] == 1
    assert out1["global_n_samples"] == 1

    state, out2 = detector.update(state, np.array([0.0]))
    assert out2["n_samples"] == 2
    assert out2["global_n_samples"] == 2


def test_manual_reset_preserve_offset_true_keeps_global_count():
    """Check that manual reset with offset preservation resets local time only."""
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=1e6)
    state = detector.init_state()

    for _ in range(5):
        state, _ = detector.update(state, np.array([0.0]))

    assert state.n_samples == 5
    assert state.n_samples_offset == 0

    state = detector.reset_state(state, preserve_offset=True)
    assert state.n_samples == 0
    assert state.n_samples_offset == 5
    assert state.grid == []
    assert state.candidate_score_states == []

    state, out = detector.update(state, np.array([0.0]))
    assert out["n_samples"] == 1
    assert out["global_n_samples"] == 6


def test_manual_reset_preserve_offset_false_restarts_global_count():
    """Check that manual reset without offset preservation restarts both clocks."""
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=1e6)
    state = detector.init_state()

    for _ in range(4):
        state, _ = detector.update(state, np.array([0.0]))

    state = detector.reset_state(state, preserve_offset=False)
    assert state.n_samples == 0
    assert state.n_samples_offset == 0

    state, out = detector.update(state, np.array([0.0]))
    assert out["n_samples"] == 1
    assert out["global_n_samples"] == 1


def test_auto_reset_on_alarm_resets_local_keeps_global():
    """Check that auto-reset returns a reset local state while preserving global time."""
    detector = GridDetector(
        score=MeanCUSUM(n_features=1),
        threshold=1.0e-12,
        auto_reset_on_alarm=True,
        preserve_offset_on_auto_reset=True,
    )
    state = detector.init_state()

    state, out1 = detector.update(state, np.array([0.0]))
    assert not out1["alarm"]
    assert out1["n_samples"] == 1
    assert out1["global_n_samples"] == 1

    state, out2 = detector.update(state, np.array([1000.0]))
    assert out2["alarm"]
    assert out2["n_samples"] == 0
    assert out2["global_n_samples"] == 2
    assert state.n_samples == 0
    assert state.n_samples_offset == 2


def test_detector_passes_global_count_to_penalty_time():
    """Check that the detector passes cumulative time into penalty scaling."""
    score = _SpyScore()
    detector = GridDetector(score=score, threshold=1e9)

    state = detector.init_state()
    for _ in range(6):
        state, _ = detector.update(state, np.array([1.0]))

    state = detector.reset_state(state, preserve_offset=True)
    assert state.n_samples_offset == 6

    state, _ = detector.update(state, np.array([1.0]))
    state, out = detector.update(state, np.array([1.0]))

    assert score.last_penalty_n_samples == 8
    assert float(out["max_score"]) == 8.0


def test_legacy_custom_score_raises_when_penalty_time_argument_is_missing():
    """Check that legacy custom scores fail explicitly when the new keyword is missing."""

    @dataclass(slots=True)
    class _OldStyleState:
        n_samples: int = 0

    @dataclass(slots=True)
    class _OldStyleScore:
        n_features: int = 1

        def init_state(self) -> _OldStyleState:
            return _OldStyleState()

        def update(self, state: _OldStyleState, x) -> _OldStyleState:
            return _OldStyleState(n_samples=state.n_samples + 1)

        def compute_penalised_scores(
            self,
            state: _OldStyleState,
            grid_states: list[_OldStyleState],
        ) -> np.ndarray:
            return np.zeros(len(grid_states), dtype=np.float64)

    detector = GridDetector(score=_OldStyleScore(), threshold=1.0)
    state = detector.init_state()

    state, _ = detector.update(state, np.array([0.0]))
    with pytest.raises(TypeError, match="n_samples_for_penalty"):
        detector.update(state, np.array([0.0]))


def test_mean_cusum_without_auto_reset_detects_shift_and_keeps_counts_running():
    """Check that plain MeanCUSUM detects the shift without resetting its clocks."""
    stream = _piecewise_constant_stream()
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=1.0)

    state, outputs = _run_stream(detector, stream)

    alarm_indices = [i for i, out in enumerate(outputs) if out["alarm"]]
    assert alarm_indices
    assert alarm_indices[0] >= 50
    assert not any(out["alarm"] for out in outputs[:50])
    assert state.n_samples == 100
    assert state.n_samples_offset == 0
    assert outputs[-1]["n_samples"] == 100
    assert outputs[-1]["global_n_samples"] == 100


def test_mean_cusum_auto_reset_with_offset_keeps_global_time_running():
    """Check that auto-reset with offset preservation resets local time but not global time."""
    stream = _piecewise_constant_stream()
    detector = GridDetector(
        score=MeanCUSUM(n_features=1),
        threshold=1.0,
        auto_reset_on_alarm=True,
        preserve_offset_on_auto_reset=True,
    )

    state, outputs, alarm_index = _run_until_first_alarm(detector, stream)

    assert alarm_index is not None
    assert alarm_index >= 50
    alarm_out = outputs[alarm_index]
    assert alarm_out["alarm"]
    assert alarm_out["n_samples"] == 0
    assert alarm_out["global_n_samples"] == alarm_index + 1
    assert state.n_samples == 0
    assert state.n_samples_offset == alarm_index + 1

    state, next_out = detector.update(state, stream[alarm_index + 1])
    assert next_out["n_samples"] == 1
    assert next_out["global_n_samples"] == alarm_index + 2


def test_mean_cusum_auto_reset_without_offset_restarts_global_time():
    """Check that auto-reset without offset preservation restarts global time too."""
    stream = _piecewise_constant_stream()
    detector = GridDetector(
        score=MeanCUSUM(n_features=1),
        threshold=1.0,
        auto_reset_on_alarm=True,
        preserve_offset_on_auto_reset=False,
    )

    state, outputs, alarm_index = _run_until_first_alarm(detector, stream)

    assert alarm_index is not None
    assert alarm_index >= 50
    alarm_out = outputs[alarm_index]
    assert alarm_out["alarm"]
    assert alarm_out["n_samples"] == 0
    assert alarm_out["global_n_samples"] == 0
    assert state.n_samples == 0
    assert state.n_samples_offset == 0

    state, next_out = detector.update(state, stream[alarm_index + 1])
    assert next_out["n_samples"] == 1
    assert next_out["global_n_samples"] == 1


def test_mean_cusum_manual_reset_with_offset_keeps_global_time_running():
    """Check that manual reset with offset preservation continues the global clock."""
    stream = _piecewise_constant_stream()
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=1.0)

    state, outputs, alarm_index = _run_until_first_alarm(detector, stream)

    assert alarm_index is not None
    assert alarm_index >= 50
    assert outputs[alarm_index]["global_n_samples"] == alarm_index + 1

    state = detector.reset_state(state, preserve_offset=True)
    assert state.n_samples == 0
    assert state.n_samples_offset == alarm_index + 1

    state, next_out = detector.update(state, stream[alarm_index + 1])
    assert next_out["n_samples"] == 1
    assert next_out["global_n_samples"] == alarm_index + 2


def test_mean_cusum_manual_reset_without_offset_restarts_global_time():
    """Check that manual reset without offset preservation restarts the global clock."""
    stream = _piecewise_constant_stream()
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=1.0)

    state, outputs, alarm_index = _run_until_first_alarm(detector, stream)

    assert alarm_index is not None
    assert alarm_index >= 50
    assert outputs[alarm_index]["global_n_samples"] == alarm_index + 1

    state = detector.reset_state(state, preserve_offset=False)
    assert state.n_samples == 0
    assert state.n_samples_offset == 0

    state, next_out = detector.update(state, stream[alarm_index + 1])
    assert next_out["n_samples"] == 1
    assert next_out["global_n_samples"] == 1
