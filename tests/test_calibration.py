import numpy as np
import pytest
import warnings

from gridcp.calibration import (
    calibrate_detector_threshold_false_alarm,
    calibrate_threshold_false_alarm,
    draw_samples,
    mc_alarm_times,
    mc_max_scores,
    with_calibrated_threshold,
)
from gridcp.detector import GridDetector
from gridcp.scores import CUSUM


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
    """Check fixed changepoint sampling with scalar pre/post samplers."""
    rng = np.random.default_rng(123)

    X = draw_samples(
        n_paths=5,
        stream_len=10,
        n_features=3,
        pre_sampler=lambda rng: 0.0,
        post_sampler=lambda rng: 10.0,
        changepoint=6,
        rng=rng,
    )

    assert X.shape == (5, 10, 3)
    assert np.allclose(X[:, :6, :], 0.0)
    assert np.allclose(X[:, 6:, :], 10.0)


def test_draw_samples_changepoint_is_first_post_change_index():
    """Verify changepoint is interpreted as first post-change index."""
    # changepoint=1 means sample at index 0 is pre-change and index >=1 post-change.
    X = draw_samples(
        n_paths=3,
        stream_len=5,
        n_features=1,
        pre_sampler=lambda rng: -2.0,
        post_sampler=lambda rng: 7.0,
        changepoint=1,
        rng=123,
    )

    assert np.allclose(X[:, 0, :], -2.0)
    assert np.allclose(X[:, 1:, :], 7.0)


def test_draw_samples_random_changepoint_callable():
    """Ensure callable changepoints produce valid sampled paths."""
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
        pre_sampler=lambda rng: -1.0,
        post_sampler=lambda rng: 2.0,
        changepoint=cp_sampler,
        rng=rng,
    )

    assert X.shape == (20, 12, 1)
    # At least one post-change value should appear across the batch.
    assert np.any(X == 2.0)


def test_draw_samples_rejects_changepoint_callable_wrong_arity():
    """Reject changepoint callable without required path_index argument."""
    with pytest.raises(TypeError, match="must accept arguments"):
        draw_samples(
            n_paths=3,
            stream_len=10,
            n_features=1,
            pre_sampler=lambda rng: -1.0,
            post_sampler=lambda rng: 2.0,
            changepoint=lambda rng, stream_len: 4,
            rng=0,
        )


def test_mc_alarm_times_rejects_changepoint_callable_extra_required_arg():
    """Reject changepoint callable requiring extra mandatory arguments."""

    def bad_cp(rng, stream_len, path_index, extra):
        return 3

    detector = GridDetector(score=CUSUM(n_features=1), threshold=10.0)

    with pytest.raises(TypeError, match="must accept arguments"):
        mc_alarm_times(
            detector=detector,
            n_paths=4,
            stream_len=12,
            pre_sampler=lambda rng: 0.0,
            post_sampler=lambda rng: 1.0,
            changepoint=bad_cp,
            rng=0,
            parallel=False,
        )


def test_draw_samples_rejects_changepoint_callable_non_integer_return():
    """Reject changepoint callable that returns a non-integer-like value."""

    def bad_cp(rng, stream_len, path_index):
        return "not-an-int"

    with pytest.raises(TypeError, match="callable returning an integer"):
        draw_samples(
            n_paths=3,
            stream_len=10,
            n_features=1,
            pre_sampler=lambda rng: -1.0,
            post_sampler=lambda rng: 2.0,
            changepoint=bad_cp,
            rng=0,
        )


def test_mc_max_scores_rejects_changepoint_callable_out_of_range_return():
    """Reject changepoint callable that returns values outside [0, stream_len]."""

    def bad_cp(rng, stream_len, path_index):
        return stream_len + 1

    detector = GridDetector(score=CUSUM(n_features=1), threshold=10.0)

    with pytest.raises(ValueError, match="changepoint must be in"):
        mc_max_scores(
            detector=detector,
            n_paths=4,
            stream_len=12,
            pre_sampler=lambda rng: 0.0,
            post_sampler=lambda rng: 1.0,
            changepoint=bad_cp,
            rng=0,
            parallel=False,
        )


def test_draw_samples_accepts_flattenable_array_output_for_multivariate():
    """Accept flattenable array-like sampler outputs in multivariate mode."""
    X = draw_samples(
        n_paths=4,
        stream_len=6,
        n_features=3,
        pre_sampler=lambda rng: np.array([[1.0, 2.0, 3.0]]),
        rng=7,
    )

    assert X.shape == (4, 6, 3)
    assert np.allclose(X, np.array([1.0, 2.0, 3.0]))


def test_draw_samples_passes_rng_positionally_for_positional_only_sampler():
    """Pass rng positionally for positional-only sampler signatures."""

    def sampler(rng: np.random.Generator, /, mean: float) -> float:
        return float(mean + rng.normal(0.0, 1.0))

    seed = 321
    expected_rng = np.random.default_rng(seed)
    expected = np.array(
        [[sampler(expected_rng, 1.5)] for _ in range(4)],
        dtype=np.float64,
    ).reshape(4, 1, 1)

    X = draw_samples(
        n_paths=4,
        stream_len=1,
        n_features=1,
        pre_sampler=sampler,
        pre_args=(1.5,),
        rng=seed,
    )

    assert np.allclose(X, expected)


def test_draw_samples_passes_rng_by_keyword_when_not_first_parameter():
    """Pass rng by keyword for signatures like sampler(mean, rng)."""

    def sampler(mean: float, rng: np.random.Generator) -> float:
        return float(mean + rng.normal(0.0, 1.0))

    seed = 1234
    expected_rng = np.random.default_rng(seed)
    expected = np.array(
        [[sampler(1.5, expected_rng)] for _ in range(4)],
        dtype=np.float64,
    ).reshape(4, 1, 1)

    X = draw_samples(
        n_paths=4,
        stream_len=1,
        n_features=1,
        pre_sampler=sampler,
        pre_args=(1.5,),
        rng=seed,
    )

    assert np.allclose(X, expected)


def test_draw_samples_equivalent_rng_signatures_produce_identical_samples():
    """Equivalent sampler signatures should produce identical draws."""

    def sampler_positional(rng: np.random.Generator, /, mean: float) -> float:
        return float(mean + rng.normal(0.0, 1.0))

    def sampler_keyword(mean: float, rng: np.random.Generator) -> float:
        return float(mean + rng.normal(0.0, 1.0))

    kwargs = dict(
        n_paths=6,
        stream_len=4,
        n_features=1,
        pre_args=(1.25,),
        rng=2026,
    )
    X_positional = draw_samples(pre_sampler=sampler_positional, **kwargs)
    X_keyword = draw_samples(pre_sampler=sampler_keyword, **kwargs)

    assert np.allclose(X_positional, X_keyword)


def test_draw_samples_warns_on_runtime_size_one_broadcast_in_multivariate():
    """Warn when a size-1 vector-like output is broadcast to multivariate shape."""

    class _Sampler:
        def __init__(self):
            self.calls = 0

        def __call__(self, rng: np.random.Generator):
            self.calls += 1
            # First call is used by preflight. Ensure first runtime draw is vector,
            # then return a size-1 array to trigger runtime broadcast warning.
            if self.calls <= 2:
                return np.array([1.0, 2.0, 3.0], dtype=np.float64)
            return np.array([9.0], dtype=np.float64)

    sampler = _Sampler()
    with pytest.warns(UserWarning, match="broadcast"):
        X = draw_samples(
            n_paths=1,
            stream_len=3,
            n_features=3,
            pre_sampler=sampler,
            rng=7,
        )

    assert X.shape == (1, 3, 3)
    assert np.allclose(X[0, 1, :], 9.0)


def test_draw_samples_requires_post_sampler_if_changepoint_set():
    """Require post_sampler whenever changepoint is provided."""
    with pytest.raises(ValueError):
        draw_samples(
            n_paths=2,
            stream_len=5,
            n_features=1,
            pre_sampler=lambda rng: 0.0,
            changepoint=3,
        )


def test_mc_max_scores_requires_post_sampler_if_changepoint_set():
    """Require post_sampler whenever changepoint is provided in mc_max_scores."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=10.0)

    with pytest.raises(ValueError, match="post_sampler must be provided"):
        mc_max_scores(
            detector=detector,
            n_paths=5,
            stream_len=10,
            pre_sampler=lambda rng: 0.0,
            changepoint=3,
            parallel=False,
        )


def test_mc_alarm_times_requires_post_sampler_if_changepoint_set():
    """Require post_sampler whenever changepoint is provided in mc_alarm_times."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=10.0)

    with pytest.raises(ValueError, match="post_sampler must be provided"):
        mc_alarm_times(
            detector=detector,
            n_paths=5,
            stream_len=10,
            pre_sampler=lambda rng: 0.0,
            changepoint=3,
            parallel=False,
        )


def test_mc_max_scores_returns_one_value_per_path():
    """Return one finite max-score value per Monte Carlo path."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=100.0)

    max_scores = mc_max_scores(
        detector=detector,
        n_paths=25,
        stream_len=30,
        pre_sampler=lambda rng: float(rng.normal(0.0, 1.0)),
        parallel=False,
    )

    assert max_scores.shape == (25, 1)
    assert np.all(np.isfinite(max_scores))


def test_mc_max_scores_accepts_array_like_sampler_for_multivariate():
    """Support array-like sampler outputs for multivariate max-score simulation."""
    detector = GridDetector(score=CUSUM(n_features=2), threshold=100.0)

    out = mc_max_scores(
        detector=detector,
        n_paths=15,
        stream_len=20,
        pre_sampler=lambda rng: [
            float(rng.normal(0.0, 1.0)),
            float(rng.normal(0.0, 1.0)),
        ],
        rng=42,
        parallel=False,
    )

    assert out.shape == (15, 1)
    assert np.all(np.isfinite(out))


def test_mc_alarm_times_returns_valid_indices_with_alarm():
    """Produce valid alarm indices and trigger alarms under strong change."""
    rng = np.random.default_rng(9)
    detector = GridDetector(score=CUSUM(n_features=1), threshold=0.1)

    alarm_times = mc_alarm_times(
        detector=detector,
        n_paths=12,
        stream_len=25,
        pre_sampler=lambda rng: 0.0,
        post_sampler=lambda rng: 8.0,
        changepoint=2,
        rng=rng,
        parallel=False,
    )

    assert alarm_times.shape == (12,)
    assert np.all((alarm_times >= 0) & (alarm_times <= 25))
    assert np.any(alarm_times < 25)


def test_mc_alarm_times_uses_stream_len_for_no_alarm():
    """Use stream_len sentinel when no alarm occurs in a path."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=1e9)

    alarm_times = mc_alarm_times(
        detector=detector,
        n_paths=10,
        stream_len=20,
        pre_sampler=lambda rng: 0.0,
        parallel=False,
    )

    assert alarm_times.shape == (10,)
    assert np.all(alarm_times == 20)


def test_mc_alarm_times_is_zero_indexed():
    """Alarm times are 0-based indices into the data array."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=0.5)

    alarm_times = mc_alarm_times(
        detector=detector,
        n_paths=5,
        stream_len=10,
        pre_sampler=lambda rng: 0.0,
        post_sampler=lambda rng: 5.0,
        changepoint=3,
        rng=42,
        parallel=False,
    )

    assert alarm_times.shape == (5,)
    # All alarms should be in [0, 10) or equal to 10 (no alarm)
    assert np.all((alarm_times >= 0) & (alarm_times <= 10))


def test_calibrate_threshold_and_with_calibrated_threshold():
    """Calibrate a threshold and verify detector wrapper preserves original detector."""
    score = CUSUM(n_features=1)
    detector = GridDetector(score=score, threshold=1.0)

    threshold = calibrate_threshold_false_alarm(
        score,
        false_alarm_probability=0.1,
        n_paths=30,
        stream_len=25,
        pre_sampler=lambda rng: float(rng.normal(0.0, 1.0)),
        parallel=False,
    )

    assert threshold.shape == (1,)
    assert np.all(np.isfinite(threshold))
    assert np.all(threshold > 0.0)

    calibrated = with_calibrated_threshold(detector, threshold)
    np.testing.assert_array_equal(calibrated.threshold, threshold)
    np.testing.assert_array_equal(detector.threshold, np.array([1.0]))


def test_calibrate_detector_threshold_wrapper_matches_score_first():
    """Check detector-first and score-first calibration APIs agree."""
    score = CUSUM(n_features=1)
    detector = GridDetector(score=score, threshold=2.0)

    threshold_from_score = calibrate_threshold_false_alarm(
        score,
        false_alarm_probability=0.1,
        n_paths=24,
        stream_len=20,
        pre_sampler=lambda rng: float(rng.normal(0.0, 1.0)),
        parallel=False,
    )

    threshold_from_detector = calibrate_detector_threshold_false_alarm(
        detector,
        false_alarm_probability=0.1,
        n_paths=24,
        stream_len=20,
        pre_sampler=lambda rng: float(rng.normal(0.0, 1.0)),
        parallel=False,
    )

    assert np.allclose(threshold_from_score, threshold_from_detector)


def test_calibrate_detector_threshold_depends_on_score_not_threshold_field():
    """Calibration wrapper should depend on detector.score, not detector.threshold."""
    score = CUSUM(n_features=1)

    detector_low_threshold = GridDetector(score=score, threshold=0.5)
    detector_high_threshold = GridDetector(score=score, threshold=5.0)

    th_low = calibrate_detector_threshold_false_alarm(
        detector_low_threshold,
        false_alarm_probability=0.1,
        n_paths=24,
        stream_len=20,
        pre_sampler=normal_sampler,
        rng=12345,
        parallel=False,
    )
    th_high = calibrate_detector_threshold_false_alarm(
        detector_high_threshold,
        false_alarm_probability=0.1,
        n_paths=24,
        stream_len=20,
        pre_sampler=normal_sampler,
        rng=12345,
        parallel=False,
    )

    assert np.allclose(th_low, th_high)


def test_mc_max_scores_accepts_int_seed_and_is_reproducible():
    """Ensure mc_max_scores is reproducible with an integer seed."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=100.0)

    out1 = mc_max_scores(
        detector=detector,
        n_paths=16,
        stream_len=12,
        pre_sampler=normal_sampler,
        rng=123,
    )
    out2 = mc_max_scores(
        detector=detector,
        n_paths=16,
        stream_len=12,
        pre_sampler=normal_sampler,
        rng=123,
    )

    assert np.allclose(out1, out2)


def test_mc_alarm_times_none_rng_is_deterministic_default():
    """Ensure rng=None yields deterministic default behavior for alarm times."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=0.7)

    out1 = mc_alarm_times(
        detector=detector,
        n_paths=16,
        stream_len=20,
        pre_sampler=normal_sampler,
        rng=None,
    )
    out2 = mc_alarm_times(
        detector=detector,
        n_paths=16,
        stream_len=20,
        pre_sampler=normal_sampler,
        rng=None,
    )

    assert np.array_equal(out1, out2)


def test_calibrate_threshold_accepts_int_seed_and_is_reproducible():
    """Ensure threshold calibration is reproducible with an integer seed."""
    score = CUSUM(n_features=1)

    th1 = calibrate_threshold_false_alarm(
        score,
        false_alarm_probability=0.1,
        n_paths=16,
        stream_len=15,
        pre_sampler=normal_sampler,
        rng=456,
    )
    th2 = calibrate_threshold_false_alarm(
        score,
        false_alarm_probability=0.1,
        n_paths=16,
        stream_len=15,
        pre_sampler=normal_sampler,
        rng=456,
    )

    assert np.array_equal(th1, th2)


def test_mc_max_scores_invalid_rng_type_raises():
    """Raise TypeError when rng has an invalid type."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=1.0)

    with pytest.raises(TypeError):
        mc_max_scores(
            detector=detector,
            n_paths=5,
            stream_len=10,
            pre_sampler=normal_sampler,
            rng="bad-rng",
        )


def test_draw_samples_rejects_sampler_without_rng_argument():
    """Reject samplers that do not accept the simulation rng."""
    bad_sampler = np.random.default_rng(10).normal

    with pytest.raises(TypeError, match="must accept an ``rng`` argument"):
        draw_samples(
            n_paths=4,
            stream_len=6,
            n_features=1,
            pre_sampler=bad_sampler,
            rng=123,
        )


def test_calibration_rejects_sampler_without_rng_argument():
    """Reject non-wrapper samplers in false-alarm calibration APIs."""
    bad_sampler = np.random.default_rng(10).normal
    score = CUSUM(n_features=1)

    with pytest.raises(TypeError, match="must accept an ``rng`` argument"):
        calibrate_threshold_false_alarm(
            score=score,
            false_alarm_probability=0.1,
            n_paths=8,
            stream_len=10,
            pre_sampler=bad_sampler,
            rng=123,
            parallel=False,
        )


def test_rejects_sampler_kwargs_containing_rng_key():
    """Reject user-provided sampler kwargs overriding internal rng handling."""
    score = CUSUM(n_features=1)

    with pytest.raises(ValueError, match="must not contain 'rng'"):
        calibrate_threshold_false_alarm(
            score=score,
            false_alarm_probability=0.1,
            n_paths=8,
            stream_len=10,
            pre_sampler=normal_sampler,
            pre_kwargs={"rng": np.random.default_rng(0)},
            rng=123,
            parallel=False,
        )


def test_mc_max_scores_parallel_reproducible_with_int_seed():
    """Ensure parallel max-score simulation is reproducible with fixed seed."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=100.0)

    out1 = mc_max_scores(
        detector=detector,
        n_paths=24,
        stream_len=25,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=20,
        rng=2026,
        parallel=True,
        n_jobs=2,
    )
    out2 = mc_max_scores(
        detector=detector,
        n_paths=24,
        stream_len=25,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=20,
        rng=2026,
        parallel=True,
        n_jobs=2,
    )

    assert np.allclose(out1, out2)


def test_mc_alarm_times_parallel_reproducible_with_int_seed():
    """Ensure parallel alarm-time simulation is reproducible with fixed seed."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=0.7)

    out1 = mc_alarm_times(
        detector=detector,
        n_paths=24,
        stream_len=25,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=random_cp_sampler,
        rng=77,
        parallel=True,
        n_jobs=2,
    )
    out2 = mc_alarm_times(
        detector=detector,
        n_paths=24,
        stream_len=25,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=random_cp_sampler,
        rng=77,
        parallel=True,
        n_jobs=2,
    )

    assert np.array_equal(out1, out2)


def test_mc_max_scores_strict_equivalence_parallel_matches_serial():
    """Check strict-equivalence mode matches serial max-score output exactly."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=100.0)

    serial = mc_max_scores(
        detector=detector,
        n_paths=24,
        stream_len=25,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=random_cp_sampler,
        rng=555,
        parallel=False,
        strict_equivalence=True,
    )
    parallel = mc_max_scores(
        detector=detector,
        n_paths=24,
        stream_len=25,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=random_cp_sampler,
        rng=555,
        parallel=True,
        n_jobs=2,
        strict_equivalence=True,
    )

    assert np.array_equal(serial, parallel)


def test_mc_alarm_times_strict_equivalence_parallel_matches_serial():
    """Check strict-equivalence mode matches serial alarm-time output exactly."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=0.6)

    serial = mc_alarm_times(
        detector=detector,
        n_paths=24,
        stream_len=25,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=random_cp_sampler,
        rng=556,
        parallel=False,
        strict_equivalence=True,
    )
    parallel = mc_alarm_times(
        detector=detector,
        n_paths=24,
        stream_len=25,
        pre_sampler=normal_sampler,
        post_sampler=shifted_normal_sampler,
        changepoint=random_cp_sampler,
        rng=556,
        parallel=True,
        n_jobs=2,
        strict_equivalence=True,
    )

    assert np.array_equal(serial, parallel)


def test_mc_max_scores_parallel_handles_local_sampler_function():
    """Confirm local sampler callables work in parallel max-score simulation."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=100.0)

    def local_sampler(rng: np.random.Generator) -> float:
        return float(rng.normal(0.0, 1.0))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = mc_max_scores(
            detector=detector,
            n_paths=20,
            stream_len=20,
            pre_sampler=local_sampler,
            rng=999,
            parallel=True,
            n_jobs=2,
        )

    assert out.shape == (20, 1)
    assert np.all(np.isfinite(out))
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_mc_alarm_times_parallel_handles_local_sampler_function():
    """Confirm local sampler callables work in parallel alarm-time simulation."""
    detector = GridDetector(score=CUSUM(n_features=1), threshold=0.6)

    def local_pre_sampler(rng: np.random.Generator) -> float:
        return float(rng.normal(0.0, 1.0))

    def local_post_sampler(rng: np.random.Generator) -> float:
        return float(rng.normal(2.0, 1.0))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = mc_alarm_times(
            detector=detector,
            n_paths=20,
            stream_len=20,
            pre_sampler=local_pre_sampler,
            post_sampler=local_post_sampler,
            changepoint=15,
            rng=12345,
            parallel=True,
            n_jobs=2,
        )

    assert out.shape == (20,)
    assert np.all((out >= 0) & (out <= 20))
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
