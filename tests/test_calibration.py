import numpy as np
import pytest
import warnings

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


def normal_sampler(rng: np.random.Generator) -> float:
    return float(rng.normal(0.0, 1.0))


def shifted_normal_sampler(rng: np.random.Generator) -> float:
    return float(rng.normal(1.5, 1.0))


def random_cp_sampler(
    local_rng: np.random.Generator,
    stream_len: int,
    _path_idx: int,
) -> int:
    low = max(2, stream_len // 4)
    high = max(low + 1, (3 * stream_len) // 4)
    return int(local_rng.integers(low=low, high=high))


def test_draw_samples_fixed_changepoint_with_scalar_sampler():
    rng = np.random.default_rng(123)

    X = draw_samples(
        n_paths=5,
        stream_len=10,
        n_features=3,
        pre_sampler=lambda: 0.0,
        post_sampler=lambda: 10.0,
        changepoint=6,
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
        parallel=False,
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
        parallel=False,
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
        parallel=False,
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
        parallel=False,
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
        parallel=False,
    )

    rng = np.random.default_rng(321)
    threshold_from_detector = calibrate_detector_threshold(
        detector,
        alpha=0.1,
        n_paths=40,
        stream_len=30,
        pre_sampler=lambda: float(rng.normal(0.0, 1.0)),
        n_features=1,
        parallel=False,
    )

    assert np.isclose(threshold_from_score, threshold_from_detector)


def test_mc_max_scores_accepts_int_seed_and_is_reproducible():
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=100.0)

    out1 = mc_max_scores(
        detector=detector,
        n_paths=30,
        stream_len=20,
        pre_sampler=normal_sampler,
        rng=123,
        n_features=1,
    )
    out2 = mc_max_scores(
        detector=detector,
        n_paths=30,
        stream_len=20,
        pre_sampler=normal_sampler,
        rng=123,
        n_features=1,
    )

    assert np.allclose(out1, out2)


def test_mc_alarm_times_none_rng_is_deterministic_default():
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=0.7)

    out1 = mc_alarm_times(
        detector=detector,
        n_paths=25,
        stream_len=30,
        pre_sampler=normal_sampler,
        rng=None,
        n_features=1,
    )
    out2 = mc_alarm_times(
        detector=detector,
        n_paths=25,
        stream_len=30,
        pre_sampler=normal_sampler,
        rng=None,
        n_features=1,
    )

    assert np.array_equal(out1, out2)


def test_calibrate_threshold_accepts_int_seed_and_is_reproducible():
    score = MeanCUSUM(n_features=1)

    th1 = calibrate_threshold(
        score,
        alpha=0.1,
        n_paths=30,
        stream_len=25,
        pre_sampler=normal_sampler,
        rng=456,
        n_features=1,
    )
    th2 = calibrate_threshold(
        score,
        alpha=0.1,
        n_paths=30,
        stream_len=25,
        pre_sampler=normal_sampler,
        rng=456,
        n_features=1,
    )

    assert th1 == th2


def test_mc_max_scores_invalid_rng_type_raises():
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=1.0)

    with pytest.raises(TypeError):
        mc_max_scores(
            detector=detector,
            n_paths=5,
            stream_len=10,
            pre_sampler=normal_sampler,
            rng="bad-rng",
            n_features=1,
        )


def test_mc_max_scores_parallel_reproducible_with_int_seed():
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=100.0)

    out1 = mc_max_scores(
        detector=detector,
        n_paths=48,
        stream_len=40,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=20,
        rng=2026,
        n_features=1,
        parallel=True,
        n_jobs=2,
    )
    out2 = mc_max_scores(
        detector=detector,
        n_paths=48,
        stream_len=40,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=20,
        rng=2026,
        n_features=1,
        parallel=True,
        n_jobs=2,
    )

    assert np.allclose(out1, out2)


def test_mc_alarm_times_parallel_reproducible_with_int_seed():
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=0.7)

    out1 = mc_alarm_times(
        detector=detector,
        n_paths=40,
        stream_len=45,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=random_cp_sampler,
        rng=77,
        n_features=1,
        parallel=True,
        n_jobs=2,
    )
    out2 = mc_alarm_times(
        detector=detector,
        n_paths=40,
        stream_len=45,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=random_cp_sampler,
        rng=77,
        n_features=1,
        parallel=True,
        n_jobs=2,
    )

    assert np.array_equal(out1, out2)


def test_mc_max_scores_strict_equivalence_parallel_matches_serial():
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=100.0)

    serial = mc_max_scores(
        detector=detector,
        n_paths=60,
        stream_len=50,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=random_cp_sampler,
        rng=555,
        n_features=1,
        parallel=False,
        strict_equivalence=True,
    )
    parallel = mc_max_scores(
        detector=detector,
        n_paths=60,
        stream_len=50,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=random_cp_sampler,
        rng=555,
        n_features=1,
        parallel=True,
        n_jobs=2,
        strict_equivalence=True,
    )

    assert np.array_equal(serial, parallel)


def test_mc_alarm_times_strict_equivalence_parallel_matches_serial():
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=0.6)

    serial = mc_alarm_times(
        detector=detector,
        n_paths=60,
        stream_len=50,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=random_cp_sampler,
        rng=556,
        n_features=1,
        parallel=False,
        strict_equivalence=True,
    )
    parallel = mc_alarm_times(
        detector=detector,
        n_paths=60,
        stream_len=50,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=random_cp_sampler,
        rng=556,
        n_features=1,
        parallel=True,
        n_jobs=2,
        strict_equivalence=True,
    )

    assert np.array_equal(serial, parallel)


def test_mc_max_scores_parallel_handles_local_sampler_function():
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=100.0)

    def local_sampler(rng: np.random.Generator) -> float:
        return float(rng.normal(0.0, 1.0))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = mc_max_scores(
            detector=detector,
            n_paths=30,
            stream_len=30,
            pre_sampler=local_sampler,
            rng=999,
            n_features=1,
            parallel=True,
            n_jobs=2,
        )

    assert out.shape == (30,)
    assert np.all(np.isfinite(out))
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_mc_alarm_times_parallel_handles_local_sampler_function():
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=0.6)

    def local_pre_sampler(rng: np.random.Generator) -> float:
        return float(rng.normal(0.0, 1.0))

    def local_post_sampler(rng: np.random.Generator) -> float:
        return float(rng.normal(2.0, 1.0))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = mc_alarm_times(
            detector=detector,
            n_paths=30,
            stream_len=30,
            pre_sampler=local_pre_sampler,
            post_sampler=local_post_sampler,
            changepoint=15,
            rng=12345,
            n_features=1,
            parallel=True,
            n_jobs=2,
        )

    assert out.shape == (30,)
    assert np.all((out >= 1) & (out <= 31))
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
