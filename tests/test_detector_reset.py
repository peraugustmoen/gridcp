import numpy as np
import pytest

from gridcp.detector import DetectorState, GridDetector
from gridcp.scores import CUSUM


def test_output_only_uses_local_n_samples():
    """Detector output reports local sample count only."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=1e6)
    state = detector.init_state()

    state, out1 = detector.update(state, np.array([0.0]))
    assert out1["n_samples"] == 1
    assert set(out1.keys()) == {"n_samples", "alarm", "max_score", "max_split_point"}

    state, out2 = detector.update(state, np.array([0.0]))
    assert out2["n_samples"] == 2
    assert set(out2.keys()) == {"n_samples", "alarm", "max_score", "max_split_point"}


def test_init_state_is_full_reset():
    """init_state clears all detector history and restarts local time."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=1e6)
    state = detector.init_state()

    for _ in range(6):
        state, _ = detector.update(state, np.array([0.0]))

    assert state.n_samples == 6
    assert state.grid
    assert state.previous_score_states

    state = detector.init_state()
    assert state.n_samples == 0
    assert state.grid == []
    assert state.previous_score_states == []
    assert state.current_score_state.n_samples == 0

    state, out = detector.update(state, np.array([0.0]))
    assert out["n_samples"] == 1


def test_init_state_has_no_offset_field():
    """Detector state carries no offset/global-time bookkeeping."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=1e6)
    state = detector.init_state()

    assert not hasattr(state, "n_samples_offset")

    state, _ = detector.update(state, np.array([0.0]))
    state = detector.init_state()
    assert not hasattr(state, "n_samples_offset")


def test_griddetector_has_no_internal_reset_api():
    """Check that reset behavior is exposed via init_state only."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=1.0)

    assert not hasattr(detector, "reset_state")
    assert not hasattr(detector, "auto_reset_on_alarm")


def test_detector_state_uses_new_field_names():
    """DetectorState exposes current_score_state / previous_score_states only."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=1e6)
    state = detector.init_state()
    for _ in range(4):
        state, _ = detector.update(state, np.array([0.0]))

    # New names present.
    assert hasattr(state, "current_score_state")
    assert hasattr(state, "previous_score_states")
    assert isinstance(state.previous_score_states, list)

    # Old names absent (no aliases).
    with pytest.raises(AttributeError):
        _ = state.running_score_state
    with pytest.raises(AttributeError):
        _ = state.candidate_score_states


def test_init_state_returns_fresh_state():
    """init_state(detector) returns a fresh state equivalent each call."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=1e6)
    state = detector.init_state()
    for _ in range(5):
        state, _ = detector.update(state, np.array([1.0]))

    fresh = detector.init_state()
    expected = detector.init_state()
    assert isinstance(fresh, DetectorState)
    assert fresh.n_samples == expected.n_samples == 0
    assert fresh.grid == expected.grid == []
    assert fresh.previous_score_states == expected.previous_score_states == []
    assert fresh.current_score_state.n_samples == 0


def test_detector_outputs_are_deterministic_after_rename():
    """Behavior invariance: a fixed stream yields identical, sensible outputs each run.

    Acts as a pure-rename guard: the detector still threads state through
    ``update`` correctly and alarms on a clear mean change.
    """
    rng = np.random.default_rng(0)
    pre = rng.normal(0.0, 1.0, size=200)
    post = rng.normal(6.0, 1.0, size=50)
    stream = np.concatenate([pre, post])

    def run() -> tuple[list[float], bool]:
        detector = GridDetector(score=CUSUM(n_features=1), threshold=5.0)
        state = detector.init_state()
        max_scores: list[float] = []
        alarmed = False
        for x in stream:
            state, out = detector.update(state, np.array([x]))
            max_scores.append(float(out["max_score"][0]))
            alarmed = alarmed or out["alarm"]
        return max_scores, alarmed

    scores_a, alarmed_a = run()
    scores_b, alarmed_b = run()

    assert scores_a == scores_b  # deterministic
    assert alarmed_a and alarmed_b  # detects the injected change
