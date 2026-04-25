import numpy as np

from gridcp import reset_detector_state
from gridcp.detector import GridDetector
from gridcp.scores import MeanCUSUM


def test_output_only_uses_local_n_samples():
    """Detector output reports local sample count only."""
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=1e6)
    state = detector.init_state()

    state, out1 = detector.update(state, np.array([0.0]))
    assert out1["n_samples"] == 1
    assert set(out1.keys()) == {"n_samples", "alarm", "max_score", "max_score_index"}

    state, out2 = detector.update(state, np.array([0.0]))
    assert out2["n_samples"] == 2
    assert set(out2.keys()) == {"n_samples", "alarm", "max_score", "max_score_index"}


def test_reset_detector_state_is_full_reset():
    """Reset clears all detector history and restarts local time."""
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=1e6)
    state = detector.init_state()

    for _ in range(6):
        state, _ = detector.update(state, np.array([0.0]))

    assert state.n_samples == 6
    assert state.grid
    assert state.candidate_score_states

    state = reset_detector_state(state, detector)
    assert state.n_samples == 0
    assert state.grid == []
    assert state.candidate_score_states == []
    assert state.running_score_state.n_samples == 0

    state, out = detector.update(state, np.array([0.0]))
    assert out["n_samples"] == 1


def test_reset_state_has_no_offset_field():
    """Detector state no longer carries any offset/global-time bookkeeping."""
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=1e6)
    state = detector.init_state()

    assert not hasattr(state, "n_samples_offset")

    state, _ = detector.update(state, np.array([0.0]))
    state = reset_detector_state(state, detector)
    assert not hasattr(state, "n_samples_offset")


def test_griddetector_has_no_internal_reset_api():
    """Check that reset behavior is exposed as external functions only."""
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=1.0)

    assert not hasattr(detector, "reset_state")
    assert not hasattr(detector, "auto_reset_on_alarm")


def test_reset_detector_state_is_importable_from_package_root():
    """Check that reset_detector_state is available from gridcp root import."""
    from gridcp import reset_detector_state as reset_fn

    assert callable(reset_fn)
