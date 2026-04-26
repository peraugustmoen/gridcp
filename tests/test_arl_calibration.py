"""Tests for ARL calibration functions (MC-based and bootstrap)."""

import warnings

import numpy as np
import pytest

from gridcp.calibration import (
    _compute_arl_threshold_from_max_scores,
    calibrate_detector_threshold_arl,
    calibrate_detector_threshold_arl_from_data,
    calibrate_threshold_arl,
    calibrate_threshold_arl_from_data,
    calibrate_threshold_arl_from_samples,
    draw_samples,
)
from gridcp.detector import GridDetector
from gridcp.scores import MeanCUSUM, MultivariateMeanIdentityCov


def _normal_sampler(rng):
    return rng.standard_normal()


class _CustomOpaqueScore:
    """Minimal custom score used to validate built-ins-only warning behavior."""

    def __init__(self, n_features: int = 1, *, enable_penalty: bool | None = None):
        self._n_features = n_features
        if enable_penalty is not None:
            self.enable_penalty = enable_penalty

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def n_tests(self) -> int:
        return 1

    def init_state(self):
        return {"sum": np.zeros(self._n_features, dtype=np.float64)}

    def update(self, state, x):
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        return {"sum": state["sum"] + x_arr[: self._n_features]}

    def compute_penalized_scores(self, state, grid_states):
        return np.zeros((len(grid_states), 1), dtype=np.float64)


# ---------------------------------------------------------------------------
# calibrate_threshold_arl (MC-based)
# ---------------------------------------------------------------------------


class TestCalibrateThresholdArl:
    def test_basic_returns_positive_vector(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        th = calibrate_threshold_arl(
            score=score,
            target_arl=50,
            n_paths=40,
            pre_sampler=_normal_sampler,
            rng=42,
            parallel=False,
        )
        assert isinstance(th, np.ndarray)
        assert th.shape == (1,)
        assert np.all(th > 0)

    def test_reproducible_with_same_seed(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        kwargs = dict(
            target_arl=50,
            n_paths=30,
            pre_sampler=_normal_sampler,
            rng=7,
            parallel=False,
        )
        assert calibrate_threshold_arl(score, **kwargs) == pytest.approx(
            calibrate_threshold_arl(score, **kwargs)
        )

    def test_invalid_target_arl(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        with pytest.raises(ValueError, match="target_arl"):
            calibrate_threshold_arl(
                score,
                target_arl=0,
                n_paths=10,
                pre_sampler=_normal_sampler,
                parallel=False,
            )

    def test_stationarity_warning_on_non_constant_penalty(self):
        score = MeanCUSUM(n_features=1)  # default: time-dependent penalty
        with pytest.warns(UserWarning, match="stationary"):
            calibrate_threshold_arl(
                score,
                target_arl=20,
                n_paths=10,
                pre_sampler=_normal_sampler,
                rng=0,
                parallel=False,
            )

    def test_builtin_with_enable_penalty_true_warns(self):
        score = MeanCUSUM(n_features=1, enable_penalty=True)
        with pytest.warns(UserWarning, match="stationary"):
            calibrate_threshold_arl(
                score,
                target_arl=20,
                n_paths=10,
                pre_sampler=_normal_sampler,
                rng=0,
                parallel=False,
            )

    def test_no_warning_for_constant_penalty(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        with warnings.catch_warnings():
            import warnings as _warnings

            _warnings.simplefilter("error", UserWarning)
            calibrate_threshold_arl(
                score,
                target_arl=20,
                n_paths=10,
                pre_sampler=_normal_sampler,
                rng=0,
                parallel=False,
            )

    def test_custom_score_with_enable_penalty_true_does_not_warn(self):
        score = _CustomOpaqueScore(n_features=1, enable_penalty=True)
        with warnings.catch_warnings():
            import warnings as _warnings

            _warnings.simplefilter("error", UserWarning)
            calibrate_threshold_arl(
                score,
                target_arl=20,
                n_paths=10,
                pre_sampler=_normal_sampler,
                rng=0,
                parallel=False,
            )

    def test_detector_wrapper_matches_score_first(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        detector = GridDetector(score=score, threshold=1.0)
        kwargs = dict(
            target_arl=50,
            n_paths=30,
            pre_sampler=_normal_sampler,
            rng=99,
            parallel=False,
        )
        th_score = calibrate_threshold_arl(score, **kwargs)
        th_det = calibrate_detector_threshold_arl(detector, **kwargs)
        np.testing.assert_allclose(th_score, th_det)


# ---------------------------------------------------------------------------
# _compute_arl_threshold_from_max_scores
# ---------------------------------------------------------------------------


class TestComputeArlThreshold:
    def test_scalar_is_1_over_e_quantile(self):
        max_scores = np.arange(1.0, 1001.0).reshape(-1, 1)
        th = _compute_arl_threshold_from_max_scores(max_scores)
        assert isinstance(th, np.ndarray)
        assert th.shape == (1,)
        assert th[0] == pytest.approx(float(np.quantile(max_scores[:, 0], 1.0 / np.e)))

    def test_multivariate_shape(self):
        rng = np.random.default_rng(0)
        max_scores = rng.uniform(1, 10, size=(1000, 3))
        th = _compute_arl_threshold_from_max_scores(max_scores)
        assert isinstance(th, np.ndarray)
        assert th.shape == (3,)
        assert np.all(th > 0)

    def test_multivariate_two_step_scaling(self):
        """Combined threshold must be >= per-test (1/e)-quantiles."""
        rng = np.random.default_rng(1)
        max_scores = rng.uniform(1, 10, size=(500, 2))
        th = _compute_arl_threshold_from_max_scores(max_scores)
        individual = np.array(
            [float(np.quantile(max_scores[:, k], 1.0 / np.e)) for k in range(2)]
        )
        # c >= 1 because combined_max >= each standardized component
        assert np.all(th >= individual)


# ---------------------------------------------------------------------------
# calibrate_threshold_arl_from_samples
# ---------------------------------------------------------------------------


class TestCalibrateArlFromSamples:
    def test_basic_scalar(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(50, 30, 1))
        th = calibrate_threshold_arl_from_samples(score=score, samples=samples)
        assert isinstance(th, np.ndarray)
        assert th.shape == (1,)
        assert np.all(th > 0)

    def test_wrong_ndim(self):
        score = MeanCUSUM(n_features=1)
        with pytest.raises(ValueError, match="3-D"):
            calibrate_threshold_arl_from_samples(
                score=score,
                samples=np.zeros((10, 5)),
            )

    def test_n_features_mismatch(self):
        score = MeanCUSUM(n_features=3)
        samples = np.zeros((10, 5, 2))
        with pytest.raises(ValueError, match="n_features"):
            calibrate_threshold_arl_from_samples(score=score, samples=samples)

    def test_stationarity_warning_on_non_constant_penalty(self):
        score = MeanCUSUM(n_features=1)  # default penalty is time-dependent
        samples = np.random.default_rng(0).normal(size=(20, 10, 1))
        with pytest.warns(UserWarning, match="stationary"):
            calibrate_threshold_arl_from_samples(score=score, samples=samples)

    def test_no_warning_for_constant_penalty(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        samples = np.random.default_rng(0).normal(size=(20, 10, 1))
        with warnings.catch_warnings():
            import warnings as _warnings

            _warnings.simplefilter("error", UserWarning)
            calibrate_threshold_arl_from_samples(score=score, samples=samples)

    def test_custom_score_without_enable_penalty_does_not_warn(self):
        score = _CustomOpaqueScore(n_features=1)
        samples = np.random.default_rng(0).normal(size=(20, 10, 1))
        with warnings.catch_warnings():
            import warnings as _warnings

            _warnings.simplefilter("error", UserWarning)
            calibrate_threshold_arl_from_samples(score=score, samples=samples)

    def test_multivariate_score(self):
        score = MultivariateMeanIdentityCov(n_features=2)
        rng = np.random.default_rng(7)
        samples = rng.normal(size=(30, 20, 2))
        th = calibrate_threshold_arl_from_samples(score=score, samples=samples)
        assert isinstance(th, np.ndarray)
        assert th.shape == (2,)
        assert np.all(th > 0)


# ---------------------------------------------------------------------------
# calibrate_threshold_arl_from_data
# ---------------------------------------------------------------------------


class TestCalibrateArlFromData:
    def test_basic_univariate(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        data = np.random.default_rng(0).normal(size=200)
        th = calibrate_threshold_arl_from_data(
            score=score,
            training_data=data,
            target_arl=50,
            n_paths=100,
            rng=42,
        )
        assert isinstance(th, np.ndarray)
        assert th.shape == (1,)
        assert np.all(th > 0)

    def test_basic_multivariate(self):
        score = MeanCUSUM(n_features=3, enable_penalty=False)
        data = np.random.default_rng(1).normal(size=(200, 3))
        th = calibrate_threshold_arl_from_data(
            score=score,
            training_data=data,
            target_arl=50,
            n_paths=100,
            rng=42,
        )
        assert isinstance(th, np.ndarray)
        assert th.shape == (1,)
        assert np.all(th > 0)

    def test_deterministic_with_same_rng(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        data = np.random.default_rng(0).normal(size=100)
        th1 = calibrate_threshold_arl_from_data(
            score=score, training_data=data, target_arl=30, n_paths=50, rng=99
        )
        th2 = calibrate_threshold_arl_from_data(
            score=score, training_data=data, target_arl=30, n_paths=50, rng=99
        )
        np.testing.assert_allclose(th1, th2)

    def test_auto_block_length(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        data = np.random.default_rng(0).normal(size=1000)
        th = calibrate_threshold_arl_from_data(
            score=score, training_data=data, target_arl=50, n_paths=50, rng=0
        )
        assert np.all(th > 0)

    def test_block_length_one(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        data = np.random.default_rng(0).normal(size=100)
        th = calibrate_threshold_arl_from_data(
            score=score,
            training_data=data,
            target_arl=30,
            n_paths=50,
            block_length=1,
            rng=0,
        )
        assert np.all(th > 0)

    def test_invalid_target_arl(self):
        score = MeanCUSUM(n_features=1)
        with pytest.raises(ValueError, match="target_arl"):
            calibrate_threshold_arl_from_data(
                score=score, training_data=np.ones(20), target_arl=0
            )

    def test_invalid_n_paths(self):
        score = MeanCUSUM(n_features=1)
        with pytest.raises(ValueError, match="n_paths"):
            calibrate_threshold_arl_from_data(
                score=score, training_data=np.ones(20), target_arl=10, n_paths=0
            )

    def test_invalid_block_length(self):
        score = MeanCUSUM(n_features=1)
        with pytest.raises(ValueError, match="block_length"):
            calibrate_threshold_arl_from_data(
                score=score,
                training_data=np.ones(20),
                target_arl=10,
                block_length=0,
            )

    def test_too_few_observations(self):
        score = MeanCUSUM(n_features=1)
        with pytest.raises(ValueError, match="at least 2"):
            calibrate_threshold_arl_from_data(
                score=score, training_data=np.array([1.0]), target_arl=10
            )

    def test_n_features_mismatch(self):
        score = MeanCUSUM(n_features=3)
        with pytest.raises(ValueError, match="features"):
            calibrate_threshold_arl_from_data(
                score=score, training_data=np.ones((20, 2)), target_arl=10
            )

    def test_wrong_ndim(self):
        score = MeanCUSUM(n_features=1)
        with pytest.raises(ValueError, match="ndim"):
            calibrate_threshold_arl_from_data(
                score=score, training_data=np.ones((5, 3, 2)), target_arl=10
            )

    def test_short_data_warns(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        data = np.ones(5)
        with pytest.warns(UserWarning, match="wrap around"):
            calibrate_threshold_arl_from_data(
                score=score, training_data=data, target_arl=20, n_paths=10, rng=0
            )


# ---------------------------------------------------------------------------
# calibrate_detector_threshold_arl_from_data
# ---------------------------------------------------------------------------


class TestCalibrateDetectorArlFromData:
    def test_matches_calibrate_threshold_from_data(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        detector = GridDetector(score=score, threshold=1.0)
        data = np.random.default_rng(5).normal(size=100)
        th1 = calibrate_threshold_arl_from_data(
            score=score,
            training_data=data,
            target_arl=30,
            n_paths=50,
            block_length=5,
            rng=42,
        )
        th2 = calibrate_detector_threshold_arl_from_data(
            detector=detector,
            training_data=data,
            target_arl=30,
            n_paths=50,
            block_length=5,
            rng=42,
        )
        np.testing.assert_allclose(th1, th2)


# ---------------------------------------------------------------------------
# Parity: from_samples on draw_samples output must match calibrate_threshold_arl
# ---------------------------------------------------------------------------


class TestParitySamplesVsMC:
    def test_scalar_parity(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        seed = 77
        n_paths, target_arl = 80, 30

        def pre_sampler(rng):
            return rng.normal()

        samples = draw_samples(
            n_paths=n_paths,
            stream_len=target_arl,
            n_features=1,
            pre_sampler=pre_sampler,
            rng=seed,
        )
        th_samples = calibrate_threshold_arl_from_samples(
            score=score, samples=samples, parallel=False
        )
        th_mc = calibrate_threshold_arl(
            score=score,
            target_arl=target_arl,
            n_paths=n_paths,
            pre_sampler=pre_sampler,
            rng=seed,
            parallel=False,
            strict_equivalence=False,
        )
        np.testing.assert_allclose(th_samples, th_mc)

    def test_multivariate_parity(self):
        score = MeanCUSUM(n_features=3, enable_penalty=False)
        seed = 55
        n_paths, target_arl = 60, 25

        def pre_sampler(rng):
            return rng.normal(size=3)

        samples = draw_samples(
            n_paths=n_paths,
            stream_len=target_arl,
            n_features=3,
            pre_sampler=pre_sampler,
            rng=seed,
        )
        th_samples = calibrate_threshold_arl_from_samples(
            score=score, samples=samples, parallel=False
        )
        th_mc = calibrate_threshold_arl(
            score=score,
            target_arl=target_arl,
            n_paths=n_paths,
            pre_sampler=pre_sampler,
            rng=seed,
            parallel=False,
            strict_equivalence=False,
        )
        np.testing.assert_allclose(th_samples, th_mc)


# ---------------------------------------------------------------------------
# Parallel vs serial equivalence
# ---------------------------------------------------------------------------


class TestParallelSerialEquivalence:
    def test_from_samples_parallel_matches_serial(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        samples = np.random.default_rng(42).normal(size=(80, 30, 1))
        th_serial = calibrate_threshold_arl_from_samples(
            score=score, samples=samples, parallel=False
        )
        th_parallel = calibrate_threshold_arl_from_samples(
            score=score, samples=samples, parallel=True
        )
        np.testing.assert_allclose(th_serial, th_parallel)

    def test_from_samples_parallel_multivariate(self):
        score = MeanCUSUM(n_features=2, enable_penalty=False)
        samples = np.random.default_rng(7).normal(size=(60, 25, 2))
        th_serial = calibrate_threshold_arl_from_samples(
            score=score, samples=samples, parallel=False
        )
        th_parallel = calibrate_threshold_arl_from_samples(
            score=score, samples=samples, parallel=True
        )
        np.testing.assert_allclose(th_serial, th_parallel)

    def test_from_data_parallel_matches_serial(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        data = np.random.default_rng(0).normal(size=200)
        th_serial = calibrate_threshold_arl_from_data(
            score=score,
            training_data=data,
            target_arl=40,
            n_paths=80,
            block_length=5,
            rng=42,
            parallel=False,
        )
        th_parallel = calibrate_threshold_arl_from_data(
            score=score,
            training_data=data,
            target_arl=40,
            n_paths=80,
            block_length=5,
            rng=42,
            parallel=True,
        )
        np.testing.assert_allclose(th_serial, th_parallel)

    def test_from_data_parallel_multivariate(self):
        score = MeanCUSUM(n_features=3, enable_penalty=False)
        data = np.random.default_rng(1).normal(size=(200, 3))
        th_serial = calibrate_threshold_arl_from_data(
            score=score,
            training_data=data,
            target_arl=40,
            n_paths=60,
            block_length=4,
            rng=99,
            parallel=False,
        )
        th_parallel = calibrate_threshold_arl_from_data(
            score=score,
            training_data=data,
            target_arl=40,
            n_paths=60,
            block_length=4,
            rng=99,
            parallel=True,
        )
        np.testing.assert_allclose(th_serial, th_parallel)

    def test_detector_wrapper_parallel_matches_serial(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        detector = GridDetector(score=score, threshold=1.0)
        data = np.random.default_rng(5).normal(size=150)
        th_serial = calibrate_detector_threshold_arl_from_data(
            detector=detector,
            training_data=data,
            target_arl=30,
            n_paths=60,
            block_length=5,
            rng=42,
            parallel=False,
        )
        th_parallel = calibrate_detector_threshold_arl_from_data(
            detector=detector,
            training_data=data,
            target_arl=30,
            n_paths=60,
            block_length=5,
            rng=42,
            parallel=True,
        )
        np.testing.assert_allclose(th_serial, th_parallel)


# ---------------------------------------------------------------------------
# n_jobs parameter
# ---------------------------------------------------------------------------


class TestNJobs:
    def test_from_samples_n_jobs_1(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        samples = np.random.default_rng(42).normal(size=(50, 20, 1))
        th_serial = calibrate_threshold_arl_from_samples(
            score=score, samples=samples, parallel=False
        )
        th_j1 = calibrate_threshold_arl_from_samples(
            score=score, samples=samples, parallel=True, n_jobs=1
        )
        np.testing.assert_allclose(th_serial, th_j1)

    def test_from_samples_n_jobs_2(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        samples = np.random.default_rng(42).normal(size=(50, 20, 1))
        th_serial = calibrate_threshold_arl_from_samples(
            score=score, samples=samples, parallel=False
        )
        th_j2 = calibrate_threshold_arl_from_samples(
            score=score, samples=samples, parallel=True, n_jobs=2
        )
        np.testing.assert_allclose(th_serial, th_j2)

    def test_from_data_n_jobs_2(self):
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        data = np.random.default_rng(0).normal(size=100)
        th_serial = calibrate_threshold_arl_from_data(
            score=score,
            training_data=data,
            target_arl=30,
            n_paths=50,
            rng=42,
            parallel=False,
        )
        th_j2 = calibrate_threshold_arl_from_data(
            score=score,
            training_data=data,
            target_arl=30,
            n_paths=50,
            rng=42,
            parallel=True,
            n_jobs=2,
        )
        np.testing.assert_allclose(th_serial, th_j2)


# ---------------------------------------------------------------------------
# resimulate_combined_threshold
# ---------------------------------------------------------------------------


class TestResimulateCombinedThreshold:
    def test_scalar_flag_is_noop(self):
        """K=1: flag must not trigger a second simulation; result == flag-off."""
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        common = dict(
            target_arl=50,
            n_paths=100,
            pre_sampler=_normal_sampler,
            rng=0,
            parallel=False,
        )
        th_off = calibrate_threshold_arl(
            score=score, resimulate_combined_threshold=False, **common
        )
        th_on = calibrate_threshold_arl(
            score=score, resimulate_combined_threshold=True, **common
        )
        assert isinstance(th_off, np.ndarray)
        assert isinstance(th_on, np.ndarray)
        assert th_off.shape == (1,)
        assert th_on.shape == (1,)
        np.testing.assert_allclose(th_off, th_on)

    def test_multivariate_flag_changes_result(self):
        """K>=2 with fixed seed: True vs False must yield different thresholds."""
        # MultivariateMeanIdentityCov emits a 2-component max_score (two tests).
        score = MultivariateMeanIdentityCov(n_features=4, enable_penalty=False)

        def sampler(rng):
            return rng.standard_normal(4)

        common = dict(
            score=score,
            target_arl=100,
            n_paths=200,
            pre_sampler=sampler,
            rng=0,
            parallel=False,
        )
        th_off = calibrate_threshold_arl(resimulate_combined_threshold=False, **common)
        th_on = calibrate_threshold_arl(resimulate_combined_threshold=True, **common)
        assert isinstance(th_off, np.ndarray) and th_off.shape == (2,)
        assert isinstance(th_on, np.ndarray) and th_on.shape == (2,)
        assert np.all(th_off > 0)
        assert np.all(th_on > 0)
        assert not np.allclose(th_off, th_on)

    def test_multivariate_flag_propagates_through_from_data(self):
        """calibrate_threshold_arl_from_data honors the flag the same way."""
        score = MultivariateMeanIdentityCov(n_features=3, enable_penalty=False)
        data = np.random.default_rng(0).normal(size=(400, 3))
        common = dict(
            score=score,
            training_data=data,
            target_arl=80,
            n_paths=200,
            rng=0,
            parallel=False,
        )
        th_off = calibrate_threshold_arl_from_data(
            resimulate_combined_threshold=False, **common
        )
        th_on = calibrate_threshold_arl_from_data(
            resimulate_combined_threshold=True, **common
        )
        assert isinstance(th_off, np.ndarray) and th_off.shape == (2,)
        assert isinstance(th_on, np.ndarray) and th_on.shape == (2,)
        assert np.all(th_off > 0)
        assert np.all(th_on > 0)
        assert not np.allclose(th_off, th_on)


# ---------------------------------------------------------------------------
# calibrate_detector_threshold_arl_from_samples
# ---------------------------------------------------------------------------


class TestCalibrateDetectorArlFromSamples:
    def test_matches_base_wrapper(self):
        """Detector wrapper must delegate exactly to the score-level function."""
        from gridcp.calibration import (
            calibrate_detector_threshold_arl_from_samples,
        )

        score = MeanCUSUM(n_features=1, enable_penalty=False)
        detector = GridDetector(score=score, threshold=1.0)
        samples = np.random.default_rng(0).normal(size=(50, 30, 1))

        th_score = calibrate_threshold_arl_from_samples(
            score=score, samples=samples, parallel=False
        )
        th_det = calibrate_detector_threshold_arl_from_samples(
            detector=detector, samples=samples, parallel=False
        )
        np.testing.assert_allclose(th_score, th_det)
