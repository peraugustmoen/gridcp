import numpy as np
import pytest

from gridcp.calibration import (
    calibrate_detector_threshold,
    calibrate_threshold,
    draw_samples,
    mc_alarm_times,
    mc_max_scores,
    with_calibrated_threshold,
)
from gridcp.detector import GridDetector
from gridcp.scores import MeanCUSUM


def test_draw_samples_fixed_changepoint_with_scalar_sampler():
    rng = np.random.default_rng(123)

    X = draw_samples(
        n_paths=5,
        stream_len=10,
        n_features=3,
        pre_sampler=lambda: 0.0,
        post_sampler=lambda: 10.0,
        changepoint=7,
        rng=rng,
    )

    assert X.shape == (5, 10, 3)
    assert np.allclose(X[:, :6, :], 0.0)
    assert np.allclose(X[:, 6:, :], 10.0)


def test_draw_samples_random_changepoint_callable():
    rng = np.random.default_rng(1)

    def cp_sampler(
        local_rng: np.random.Generator,
        stream_len: int,
        _path_idx: int,
    ) -> int:
        return int(local_rng.integers(low=4, high=7))

    X = draw_samples(
        n_paths=20,
        stream_len=12,
        n_features=1,
        pre_sampler=lambda: -1.0,
        post_sampler=lambda: 2.0,
        changepoint=cp_sampler,
        rng=rng,
    )

    assert X.shape == (20, 12, 1)
    # At least one post-change value should appear across the batch.
    assert np.any(X == 2.0)


def test_draw_samples_requires_post_sampler_if_changepoint_set():
    with pytest.raises(ValueError):
        draw_samples(
            n_paths=2,
            stream_len=5,
            n_features=1,
            pre_sampler=lambda: 0.0,
            changepoint=3,
        )


def test_mc_max_scores_returns_one_value_per_path():
    rng = np.random.default_rng(2)
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=100.0)

    max_scores = mc_max_scores(
        detector=detector,
        n_paths=25,
        stream_len=30,
        pre_sampler=lambda: float(rng.normal(0.0, 1.0)),
        n_features=1,
    )

    assert max_scores.shape == (25,)
    assert np.all(np.isfinite(max_scores))


def test_mc_alarm_times_returns_valid_indices_with_alarm():
    rng = np.random.default_rng(9)
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=0.1)

    alarm_times = mc_alarm_times(
        detector=detector,
        n_paths=12,
        stream_len=25,
        pre_sampler=lambda: 0.0,
        post_sampler=lambda: 8.0,
        changepoint=2,
        rng=rng,
        n_features=1,
    )

    assert alarm_times.shape == (12,)
    assert np.all((alarm_times >= 1) & (alarm_times <= 26))
    assert np.any(alarm_times <= 25)


def test_mc_alarm_times_uses_stream_len_plus_one_for_no_alarm():
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=1e9)

    alarm_times = mc_alarm_times(
        detector=detector,
        n_paths=10,
        stream_len=20,
        pre_sampler=lambda: 0.0,
        n_features=1,
    )

    assert alarm_times.shape == (10,)
    assert np.all(alarm_times == 21)


def test_calibrate_threshold_and_with_calibrated_threshold():
    rng = np.random.default_rng(1234)
    score = MeanCUSUM(n_features=1)
    detector = GridDetector(score=score, threshold=1.0)

    threshold = calibrate_threshold(
        score,
        alpha=0.1,
        n_paths=50,
        stream_len=40,
        pre_sampler=lambda: float(rng.normal(0.0, 1.0)),
        n_features=1,
    )

    assert np.isfinite(threshold)
    assert threshold > 0.0

    calibrated = with_calibrated_threshold(detector, threshold)
    assert calibrated.threshold == threshold
    assert detector.threshold == 1.0


def test_calibrate_detector_threshold_wrapper_matches_score_first():
    rng = np.random.default_rng(321)
    score = MeanCUSUM(n_features=1)
    detector = GridDetector(score=score, threshold=2.0)

    threshold_from_score = calibrate_threshold(
        score,
        alpha=0.1,
        n_paths=40,
        stream_len=30,
        pre_sampler=lambda: float(rng.normal(0.0, 1.0)),
        n_features=1,
    )

    rng = np.random.default_rng(321)
    threshold_from_detector = calibrate_detector_threshold(
        detector,
        alpha=0.1,
        n_paths=40,
        stream_len=30,
        pre_sampler=lambda: float(rng.normal(0.0, 1.0)),
        n_features=1,
    )

    assert np.isclose(threshold_from_score, threshold_from_detector)
