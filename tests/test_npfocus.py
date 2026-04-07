import numpy as np
import pytest

from gridcp import (
    calibrate_detector_threshold,
    mc_alarm_times,
    with_calibrated_threshold,
)
from gridcp.detector import GridDetector
from gridcp.scores import NPFOCuS


STREAM_LEN = 80
CHANGEPOINT = 40
CALIBRATION_PATHS = 80
EVAL_PATHS = 120
FALSE_ALARM_PROBABILITY = 0.05


def _gaussian_sampler(mean: float, std: float = 1.0):
    def sample(rng: np.random.Generator) -> float:
        return float(rng.normal(mean, std))

    return sample


def _poisson_sampler(rate: float):
    def sample(rng: np.random.Generator) -> float:
        return float(rng.poisson(rate))

    return sample


def _exponential_sampler(rate: float):
    def sample(rng: np.random.Generator) -> float:
        return float(rng.exponential(scale=1.0 / rate))

    return sample


def _calibrated_detector(grid: np.ndarray, pre_sampler):
    detector = GridDetector(
        score=NPFOCuS(grid=grid, n_features=1),
        threshold=1.0,
    )
    threshold = calibrate_detector_threshold(
        detector,
        false_alarm_probability=FALSE_ALARM_PROBABILITY,
        n_paths=CALIBRATION_PATHS,
        stream_len=STREAM_LEN,
        pre_sampler=pre_sampler,
        rng=123,
        parallel=False,
    )
    calibrated = with_calibrated_threshold(detector, threshold)
    return calibrated, float(threshold)


def _alarm_times(detector: GridDetector, pre_sampler, post_sampler=None) -> np.ndarray:
    return mc_alarm_times(
        detector=detector,
        n_paths=EVAL_PATHS,
        stream_len=STREAM_LEN,
        pre_sampler=pre_sampler,
        post_sampler=post_sampler,
        changepoint=CHANGEPOINT if post_sampler is not None else None,
        rng=456,
        parallel=False,
    )


@pytest.mark.parametrize(
    ("grid", "pre_sampler", "post_sampler"),
    [
        (
            np.linspace(-3.0, 3.0, 25),
            _gaussian_sampler(0.0),
            _gaussian_sampler(1.5),
        ),
        (
            np.linspace(0.5, 8.5, 17),
            _poisson_sampler(2.0),
            _poisson_sampler(6.0),
        ),
        (
            np.linspace(0.02, 2.0, 25),
            _exponential_sampler(1.0),
            _exponential_sampler(3.0),
        ),
    ],
    ids=["gaussian_mean_change", "poisson_rate_change", "exponential_rate_change"],
)
def test_npfocus_calibrated_detector_detects_distribution_changes(
    grid: np.ndarray,
    pre_sampler,
    post_sampler,
):
    """Calibrate NPFOCuS at alpha=0.05 and check it responds to each change."""
    detector, threshold = _calibrated_detector(grid, pre_sampler)

    assert np.isfinite(threshold)
    assert threshold > 0.0

    null_alarm_times = _alarm_times(detector, pre_sampler)
    change_alarm_times = _alarm_times(detector, pre_sampler, post_sampler)

    null_alarm_rate = np.mean(null_alarm_times < STREAM_LEN)
    change_alarm_rate = np.mean(change_alarm_times < STREAM_LEN)

    assert null_alarm_rate <= 0.15
    assert change_alarm_rate >= 0.45
    assert change_alarm_rate >= null_alarm_rate + 0.25

    change_detected = change_alarm_times[change_alarm_times < STREAM_LEN]
    assert change_detected.size > 0
    assert float(np.median(change_detected)) < CHANGEPOINT + 20
