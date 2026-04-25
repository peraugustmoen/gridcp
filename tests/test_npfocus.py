# ruff: noqa: E402

from pathlib import Path
import sys

import numpy as np
import pytest

# Running this file directly puts tests/ on sys.path, so prefer the repo root
# over any installed gridcp package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gridcp import (
    calibrate_detector_threshold_false_alarm,
    mc_alarm_times,
    with_calibrated_threshold,
)
from gridcp.detector import GridDetector
from gridcp.typing import PenaltyType
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


def _calibrated_detector(value_grid: np.ndarray, pre_sampler):
    detector = GridDetector(
        score=NPFOCuS(value_grid=value_grid, n_features=1),
        threshold=1.0,
    )
    threshold = calibrate_detector_threshold_false_alarm(
        detector,
        false_alarm_probability=FALSE_ALARM_PROBABILITY,
        n_paths=CALIBRATION_PATHS,
        stream_len=STREAM_LEN,
        pre_sampler=pre_sampler,
        rng=123,
        parallel=False,
    )
    calibrated = with_calibrated_threshold(detector, threshold)
    return calibrated, np.asarray(threshold, dtype=np.float64)


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


def test_npfocus_time_dependent_penalty_matches_exponential_family_glr_shape():
    score = NPFOCuS(
        value_grid=np.linspace(-3.0, 3.0, 25),
        n_features=1,
        penalty=PenaltyType.TIME_DEPENDENT,
    )
    t = 100
    assert np.isclose(score._get_penalty(t), np.sqrt(2.0 * np.log(t)) + np.log(t))


def test_npfocus_constant_penalty_is_one():
    score = NPFOCuS(
        value_grid=np.linspace(-3.0, 3.0, 25),
        n_features=1,
        penalty=PenaltyType.CONSTANT,
    )
    assert score._get_penalty(10) == 1.0
    assert score._get_penalty(1000) == 1.0


def test_npfocus_multivariate_update_and_score_shapes():
    score = NPFOCuS(
        value_grid=np.linspace(-2.0, 2.0, 11),
        n_features=2,
        penalty=PenaltyType.CONSTANT,
    )

    detector = GridDetector(score=score, threshold=np.array([1.0e6, 1.0e6]))
    state = detector.init_state()

    x = np.array(
        [
            [0.0, 0.5],
            [1.0, -0.5],
            [1.5, 0.25],
            [2.0, -1.0],
        ]
    )

    outputs = []
    for row in x:
        state, out = detector.update(state, row)
        outputs.append(out)

    assert state.running_score_state.n_smaller.shape == (2, 11)
    assert outputs[-1]["max_score"].shape == (2,)
    assert outputs[-1]["max_score_index"].shape == (2,)


def test_npfocus_multivariate_scores_take_max_over_channels():
    value_grid = np.array([-1.0, 0.0, 1.0])
    score_multi = NPFOCuS(
        value_grid=value_grid,
        n_features=2,
        penalty=PenaltyType.CONSTANT,
    )
    score_ch0 = NPFOCuS(
        value_grid=value_grid,
        n_features=1,
        penalty=PenaltyType.CONSTANT,
    )
    score_ch1 = NPFOCuS(
        value_grid=value_grid,
        n_features=1,
        penalty=PenaltyType.CONSTANT,
    )

    x = np.array(
        [
            [-2.0, 0.2],
            [-1.5, 0.1],
            [-0.8, -0.2],
            [2.5, 0.3],
            [2.7, 0.4],
            [3.0, 0.5],
        ]
    )

    det_multi = GridDetector(score=score_multi, threshold=np.array([1.0e6, 1.0e6]))
    det_ch0 = GridDetector(score=score_ch0, threshold=np.array([1.0e6, 1.0e6]))
    det_ch1 = GridDetector(score=score_ch1, threshold=np.array([1.0e6, 1.0e6]))

    state_multi = det_multi.init_state()
    state_ch0 = det_ch0.init_state()
    state_ch1 = det_ch1.init_state()

    for row in x:
        state_multi, _ = det_multi.update(state_multi, row)
        state_ch0, _ = det_ch0.update(state_ch0, row[0])
        state_ch1, _ = det_ch1.update(state_ch1, row[1])

    scores_multi = score_multi.compute_penalised_scores(
        state_multi.running_score_state,
        state_multi.candidate_score_states,
    )
    scores_ch0 = score_ch0.compute_penalised_scores(
        state_ch0.running_score_state,
        state_ch0.candidate_score_states,
    )
    scores_ch1 = score_ch1.compute_penalised_scores(
        state_ch1.running_score_state,
        state_ch1.candidate_score_states,
    )

    assert np.allclose(
        scores_multi[:, 0], np.maximum(scores_ch0[:, 0], scores_ch1[:, 0])
    )
    assert np.allclose(
        scores_multi[:, 1], np.maximum(scores_ch0[:, 1], scores_ch1[:, 1])
    )


@pytest.mark.parametrize(
    ("value_grid", "pre_sampler", "post_sampler"),
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
    value_grid: np.ndarray,
    pre_sampler,
    post_sampler,
):
    """Calibrate NPFOCuS at alpha=0.05 and check it responds to each change."""
    detector, threshold = _calibrated_detector(value_grid, pre_sampler)

    assert threshold.shape == (2,)
    assert np.all(np.isfinite(threshold))
    assert np.all(threshold > 0.0)

    state = detector.init_state()
    state, _ = detector.update(state, pre_sampler(np.random.default_rng(1)))
    state, output = detector.update(state, pre_sampler(np.random.default_rng(2)))
    assert isinstance(output["max_score"], np.ndarray)
    assert output["max_score"].shape == (2,)
    assert isinstance(output["max_score_index"], np.ndarray)
    assert output["max_score_index"].shape == (2,)

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
