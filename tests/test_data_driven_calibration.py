"""Tests for data-driven (bootstrap) calibration functions."""

import numpy as np
import pytest

from gridcp.calibration import (
    _block_bootstrap_samples,
    _compute_threshold_from_max_scores,
    calibrate_detector_threshold_from_data,
    calibrate_threshold,
    calibrate_threshold_from_data,
    calibrate_threshold_from_samples,
    draw_samples,
)
from gridcp.detector import GridDetector
from gridcp.scores import MeanCUSUM, MultivariateMeanIdentityCov
from gridcp.typing import PenaltyType


# ---------------------------------------------------------------------------
# _block_bootstrap_samples
# ---------------------------------------------------------------------------


class TestBlockBootstrapSamples:
    """Tests for the circular block bootstrap resampling."""

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        data = np.arange(30, dtype=np.float64).reshape(10, 3)
        out = _block_bootstrap_samples(
            data, n_paths=5, stream_len=8, block_length=3, rng=rng
        )
        assert out.shape == (5, 8, 3)

    def test_block_length_one_is_iid(self):
        """block_length=1 should resample individual observations."""
        rng = np.random.default_rng(7)
        data = np.arange(20, dtype=np.float64).reshape(20, 1)
        out = _block_bootstrap_samples(
            data, n_paths=100, stream_len=20, block_length=1, rng=rng
        )
        # Every resampled value must be one of the original values
        original_vals = set(data.ravel().tolist())
        for val in out.ravel():
            assert val in original_vals

    def test_circular_wrap(self):
        """Blocks starting near the end should wrap around."""
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        rng = np.random.default_rng(0)
        out = _block_bootstrap_samples(
            data, n_paths=500, stream_len=10, block_length=4, rng=rng
        )
        # All values must be from the original data
        original_vals = set(data.ravel().tolist())
        for val in out.ravel():
            assert val in original_vals

    def test_deterministic_with_same_seed(self):
        data = np.random.default_rng(0).normal(size=(50, 2))
        out1 = _block_bootstrap_samples(data, 10, 20, 5, np.random.default_rng(99))
        out2 = _block_bootstrap_samples(data, 10, 20, 5, np.random.default_rng(99))
        np.testing.assert_array_equal(out1, out2)

    def test_stream_len_longer_than_data(self):
        """Should work even when stream_len > N via wrap-around."""
        data = np.ones((5, 1), dtype=np.float64)
        rng = np.random.default_rng(3)
        out = _block_bootstrap_samples(
            data, n_paths=2, stream_len=20, block_length=3, rng=rng
        )
        assert out.shape == (2, 20, 1)
        np.testing.assert_allclose(out, 1.0)

    def test_block_length_equals_stream_len(self):
        """Each path is exactly one block."""
        data = np.arange(100, dtype=np.float64).reshape(100, 1)
        rng = np.random.default_rng(11)
        out = _block_bootstrap_samples(
            data, n_paths=3, stream_len=10, block_length=10, rng=rng
        )
        assert out.shape == (3, 10, 1)
        # Each path should be a contiguous (possibly wrapped) block
        for p in range(3):
            vals = out[p, :, 0]
            diffs = np.diff(vals) % 100
            np.testing.assert_array_equal(diffs, np.ones(9))


# ---------------------------------------------------------------------------
# _compute_threshold_from_max_scores
# ---------------------------------------------------------------------------


class TestComputeThreshold:
    """Tests for the shared threshold quantile helper."""

    def test_scalar_scores(self):
        max_scores = np.arange(1.0, 101.0)  # 1, 2, ..., 100
        th = _compute_threshold_from_max_scores(max_scores, 0.05, apply_bonferroni=True)
        assert isinstance(th, float)
        assert th == pytest.approx(np.quantile(max_scores, 0.95))

    def test_multivariate_with_bonferroni(self):
        rng = np.random.default_rng(0)
        max_scores = rng.normal(size=(1000, 3))
        th = _compute_threshold_from_max_scores(max_scores, 0.06, apply_bonferroni=True)
        assert isinstance(th, np.ndarray)
        assert th.shape == (3,)
        # Bonferroni: alpha_corrected = 0.06/3 = 0.02
        for k in range(3):
            expected = float(np.quantile(max_scores[:, k], 1.0 - 0.02))
            assert th[k] == pytest.approx(expected)

    def test_multivariate_without_bonferroni(self):
        rng = np.random.default_rng(1)
        max_scores = rng.normal(size=(500, 2))
        th = _compute_threshold_from_max_scores(
            max_scores, 0.10, apply_bonferroni=False
        )
        assert isinstance(th, np.ndarray)
        for k in range(2):
            expected = float(np.quantile(max_scores[:, k], 0.90))
            assert th[k] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# calibrate_threshold_from_samples
# ---------------------------------------------------------------------------


class TestCalibrateThresholdFromSamples:
    """Tests for calibrate_threshold_from_samples."""

    def test_basic_scalar(self):
        score = MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT)
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(50, 30, 1))
        th = calibrate_threshold_from_samples(
            score=score,
            samples=samples,
            false_alarm_probability=0.10,
        )
        assert isinstance(th, float)
        assert th > 0

    def test_invalid_fap(self):
        score = MeanCUSUM(n_features=1)
        samples = np.zeros((5, 10, 1))
        with pytest.raises(ValueError, match="false_alarm_probability"):
            calibrate_threshold_from_samples(
                score=score, samples=samples, false_alarm_probability=0.0
            )
        with pytest.raises(ValueError, match="false_alarm_probability"):
            calibrate_threshold_from_samples(
                score=score, samples=samples, false_alarm_probability=1.0
            )

    def test_wrong_ndim(self):
        score = MeanCUSUM(n_features=1)
        with pytest.raises(ValueError, match="3-D"):
            calibrate_threshold_from_samples(
                score=score,
                samples=np.zeros((10, 5)),
                false_alarm_probability=0.05,
            )

    def test_n_features_mismatch(self):
        score = MeanCUSUM(n_features=3)
        samples = np.zeros((10, 5, 2))
        with pytest.raises(ValueError, match="n_features"):
            calibrate_threshold_from_samples(
                score=score,
                samples=samples,
                false_alarm_probability=0.05,
            )

    def test_multivariate_score(self):
        score = MultivariateMeanIdentityCov(n_features=2)
        rng = np.random.default_rng(7)
        samples = rng.normal(size=(30, 20, 2))
        th = calibrate_threshold_from_samples(
            score=score,
            samples=samples,
            false_alarm_probability=0.10,
        )
        # MultivariateMeanIdentityCov returns scalar score per candidate
        assert isinstance(th, float) or (isinstance(th, np.ndarray) and th.ndim <= 1)
        if isinstance(th, np.ndarray):
            assert np.all(th > 0)
        else:
            assert th > 0


# ---------------------------------------------------------------------------
# calibrate_threshold_from_data
# ---------------------------------------------------------------------------


class TestCalibrateThresholdFromData:
    """Tests for calibrate_threshold_from_data."""

    def test_basic_univariate(self):
        score = MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT)
        rng = np.random.default_rng(0)
        training_data = rng.normal(size=200)
        th = calibrate_threshold_from_data(
            score=score,
            training_data=training_data,
            false_alarm_probability=0.05,
            stream_len=50,
            n_paths=100,
            rng=42,
        )
        assert isinstance(th, float)
        assert th > 0

    def test_basic_multivariate(self):
        score = MeanCUSUM(n_features=3, penalty=PenaltyType.CONSTANT)
        rng = np.random.default_rng(1)
        training_data = rng.normal(size=(200, 3))
        th = calibrate_threshold_from_data(
            score=score,
            training_data=training_data,
            false_alarm_probability=0.05,
            stream_len=50,
            n_paths=100,
            rng=42,
        )
        assert isinstance(th, float)
        assert th > 0

    def test_deterministic_with_same_rng(self):
        score = MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT)
        data = np.random.default_rng(0).normal(size=100)
        th1 = calibrate_threshold_from_data(
            score=score,
            training_data=data,
            false_alarm_probability=0.05,
            stream_len=30,
            n_paths=50,
            rng=99,
        )
        th2 = calibrate_threshold_from_data(
            score=score,
            training_data=data,
            false_alarm_probability=0.05,
            stream_len=30,
            n_paths=50,
            rng=99,
        )
        assert th1 == pytest.approx(th2)

    def test_auto_block_length(self):
        """Default block_length should be max(1, int(N^(1/3)))."""
        score = MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT)
        data = np.random.default_rng(0).normal(size=1000)
        # N=1000, N^(1/3)=10, so block_length=10
        # Just verify it runs without error
        th = calibrate_threshold_from_data(
            score=score,
            training_data=data,
            false_alarm_probability=0.05,
            stream_len=50,
            n_paths=50,
            rng=0,
        )
        assert th > 0

    def test_block_length_one(self):
        """Explicit block_length=1 should work (i.i.d. bootstrap)."""
        score = MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT)
        data = np.random.default_rng(0).normal(size=100)
        th = calibrate_threshold_from_data(
            score=score,
            training_data=data,
            false_alarm_probability=0.05,
            stream_len=30,
            n_paths=50,
            block_length=1,
            rng=0,
        )
        assert th > 0

    def test_invalid_fap(self):
        score = MeanCUSUM(n_features=1)
        data = np.ones(20)
        with pytest.raises(ValueError, match="false_alarm_probability"):
            calibrate_threshold_from_data(
                score=score,
                training_data=data,
                false_alarm_probability=1.5,
                stream_len=10,
            )

    def test_invalid_stream_len(self):
        score = MeanCUSUM(n_features=1)
        with pytest.raises(ValueError, match="stream_len"):
            calibrate_threshold_from_data(
                score=score,
                training_data=np.ones(20),
                false_alarm_probability=0.05,
                stream_len=0,
            )

    def test_invalid_n_paths(self):
        score = MeanCUSUM(n_features=1)
        with pytest.raises(ValueError, match="n_paths"):
            calibrate_threshold_from_data(
                score=score,
                training_data=np.ones(20),
                false_alarm_probability=0.05,
                stream_len=10,
                n_paths=0,
            )

    def test_invalid_block_length(self):
        score = MeanCUSUM(n_features=1)
        with pytest.raises(ValueError, match="block_length"):
            calibrate_threshold_from_data(
                score=score,
                training_data=np.ones(20),
                false_alarm_probability=0.05,
                stream_len=10,
                block_length=0,
            )

    def test_too_few_observations(self):
        score = MeanCUSUM(n_features=1)
        with pytest.raises(ValueError, match="at least 2"):
            calibrate_threshold_from_data(
                score=score,
                training_data=np.array([1.0]),
                false_alarm_probability=0.05,
                stream_len=10,
            )

    def test_n_features_mismatch(self):
        score = MeanCUSUM(n_features=3)
        data = np.ones((20, 2))
        with pytest.raises(ValueError, match="features"):
            calibrate_threshold_from_data(
                score=score,
                training_data=data,
                false_alarm_probability=0.05,
                stream_len=10,
            )

    def test_wrong_ndim(self):
        score = MeanCUSUM(n_features=1)
        with pytest.raises(ValueError, match="ndim"):
            calibrate_threshold_from_data(
                score=score,
                training_data=np.ones((5, 3, 2)),
                false_alarm_probability=0.05,
                stream_len=10,
            )

    def test_short_data_warns(self):
        score = MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT)
        data = np.ones(5)
        with pytest.warns(UserWarning, match="wrap around"):
            calibrate_threshold_from_data(
                score=score,
                training_data=data,
                false_alarm_probability=0.05,
                stream_len=20,
                n_paths=10,
                rng=0,
            )


# ---------------------------------------------------------------------------
# calibrate_detector_threshold_from_data
# ---------------------------------------------------------------------------


class TestCalibrateDetectorThresholdFromData:
    """Tests for calibrate_detector_threshold_from_data."""

    def test_matches_calibrate_threshold_from_data(self):
        score = MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT)
        detector = GridDetector(score=score, threshold=1.0)
        data = np.random.default_rng(5).normal(size=100)
        th1 = calibrate_threshold_from_data(
            score=score,
            training_data=data,
            false_alarm_probability=0.05,
            stream_len=30,
            n_paths=50,
            block_length=5,
            rng=42,
        )
        th2 = calibrate_detector_threshold_from_data(
            detector=detector,
            training_data=data,
            false_alarm_probability=0.05,
            stream_len=30,
            n_paths=50,
            block_length=5,
            rng=42,
        )
        assert th1 == pytest.approx(th2)


# ---------------------------------------------------------------------------
# Parity: data-driven vs MC calibration on the same samples
# ---------------------------------------------------------------------------


class TestParitySamplesVsMC:
    """calibrate_threshold_from_samples on draw_samples output must equal
    calibrate_threshold with strict_equivalence (both use the same samples)."""

    def test_scalar_parity(self):
        score = MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT)
        seed = 77
        n_paths, stream_len, fap = 80, 30, 0.05

        def pre_sampler(rng):
            return rng.normal()

        samples = draw_samples(
            n_paths=n_paths,
            stream_len=stream_len,
            n_features=1,
            pre_sampler=pre_sampler,
            rng=seed,
        )
        th_samples = calibrate_threshold_from_samples(
            score=score,
            samples=samples,
            false_alarm_probability=fap,
            parallel=False,
        )
        th_mc = calibrate_threshold(
            score=score,
            false_alarm_probability=fap,
            n_paths=n_paths,
            stream_len=stream_len,
            pre_sampler=pre_sampler,
            rng=seed,
            parallel=False,
            strict_equivalence=False,
        )
        assert th_samples == pytest.approx(th_mc)

    def test_multivariate_parity(self):
        score = MeanCUSUM(n_features=3, penalty=PenaltyType.CONSTANT)
        seed = 55
        n_paths, stream_len, fap = 60, 25, 0.10

        def pre_sampler(rng):
            return rng.normal(size=3)

        samples = draw_samples(
            n_paths=n_paths,
            stream_len=stream_len,
            n_features=3,
            pre_sampler=pre_sampler,
            rng=seed,
        )
        th_samples = calibrate_threshold_from_samples(
            score=score,
            samples=samples,
            false_alarm_probability=fap,
            parallel=False,
        )
        th_mc = calibrate_threshold(
            score=score,
            false_alarm_probability=fap,
            n_paths=n_paths,
            stream_len=stream_len,
            pre_sampler=pre_sampler,
            rng=seed,
            parallel=False,
            strict_equivalence=False,
        )
        np.testing.assert_allclose(th_samples, th_mc)

    def test_parity_with_parallel_strict_equivalence(self):
        """MC strict_equivalence=True must also agree since it uses draw_samples."""
        score = MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT)
        seed = 33
        n_paths, stream_len, fap = 80, 30, 0.05

        def pre_sampler(rng):
            return rng.normal()

        samples = draw_samples(
            n_paths=n_paths,
            stream_len=stream_len,
            n_features=1,
            pre_sampler=pre_sampler,
            rng=seed,
        )
        th_from_samples = calibrate_threshold_from_samples(
            score=score,
            samples=samples,
            false_alarm_probability=fap,
            parallel=False,
        )
        th_mc_strict = calibrate_threshold(
            score=score,
            false_alarm_probability=fap,
            n_paths=n_paths,
            stream_len=stream_len,
            pre_sampler=pre_sampler,
            rng=seed,
            parallel=True,
            strict_equivalence=True,
        )
        assert th_from_samples == pytest.approx(th_mc_strict)


# ---------------------------------------------------------------------------
# Parallel vs serial equivalence
# ---------------------------------------------------------------------------


class TestParallelSerialEquivalence:
    """parallel=True must produce the same threshold as parallel=False."""

    def test_from_samples_parallel_matches_serial(self):
        score = MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT)
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(80, 30, 1))
        th_serial = calibrate_threshold_from_samples(
            score=score, samples=samples, false_alarm_probability=0.05, parallel=False
        )
        th_parallel = calibrate_threshold_from_samples(
            score=score, samples=samples, false_alarm_probability=0.05, parallel=True
        )
        assert th_serial == pytest.approx(th_parallel)

    def test_from_samples_parallel_multivariate(self):
        score = MeanCUSUM(n_features=2, penalty=PenaltyType.CONSTANT)
        rng = np.random.default_rng(7)
        samples = rng.normal(size=(60, 25, 2))
        th_serial = calibrate_threshold_from_samples(
            score=score, samples=samples, false_alarm_probability=0.10, parallel=False
        )
        th_parallel = calibrate_threshold_from_samples(
            score=score, samples=samples, false_alarm_probability=0.10, parallel=True
        )
        np.testing.assert_allclose(th_serial, th_parallel)

    def test_from_data_parallel_matches_serial(self):
        score = MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT)
        data = np.random.default_rng(0).normal(size=200)
        th_serial = calibrate_threshold_from_data(
            score=score,
            training_data=data,
            false_alarm_probability=0.05,
            stream_len=40,
            n_paths=80,
            block_length=5,
            rng=42,
            parallel=False,
        )
        th_parallel = calibrate_threshold_from_data(
            score=score,
            training_data=data,
            false_alarm_probability=0.05,
            stream_len=40,
            n_paths=80,
            block_length=5,
            rng=42,
            parallel=True,
        )
        assert th_serial == pytest.approx(th_parallel)

    def test_from_data_parallel_multivariate(self):
        score = MeanCUSUM(n_features=3, penalty=PenaltyType.CONSTANT)
        data = np.random.default_rng(1).normal(size=(200, 3))
        th_serial = calibrate_threshold_from_data(
            score=score,
            training_data=data,
            false_alarm_probability=0.05,
            stream_len=40,
            n_paths=60,
            block_length=4,
            rng=99,
            parallel=False,
        )
        th_parallel = calibrate_threshold_from_data(
            score=score,
            training_data=data,
            false_alarm_probability=0.05,
            stream_len=40,
            n_paths=60,
            block_length=4,
            rng=99,
            parallel=True,
        )
        np.testing.assert_allclose(th_serial, th_parallel)

    def test_detector_wrapper_parallel_matches_serial(self):
        score = MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT)
        detector = GridDetector(score=score, threshold=1.0)
        data = np.random.default_rng(5).normal(size=150)
        th_serial = calibrate_detector_threshold_from_data(
            detector=detector,
            training_data=data,
            false_alarm_probability=0.05,
            stream_len=30,
            n_paths=60,
            block_length=5,
            rng=42,
            parallel=False,
        )
        th_parallel = calibrate_detector_threshold_from_data(
            detector=detector,
            training_data=data,
            false_alarm_probability=0.05,
            stream_len=30,
            n_paths=60,
            block_length=5,
            rng=42,
            parallel=True,
        )
        assert th_serial == pytest.approx(th_parallel)


# ---------------------------------------------------------------------------
# n_jobs parameter
# ---------------------------------------------------------------------------


class TestNJobs:
    """Verify that explicit n_jobs values produce correct results."""

    def test_from_samples_n_jobs_1(self):
        """n_jobs=1 with parallel=True should behave like serial."""
        score = MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT)
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(50, 20, 1))
        th_serial = calibrate_threshold_from_samples(
            score=score, samples=samples, false_alarm_probability=0.05, parallel=False
        )
        th_j1 = calibrate_threshold_from_samples(
            score=score,
            samples=samples,
            false_alarm_probability=0.05,
            parallel=True,
            n_jobs=1,
        )
        assert th_serial == pytest.approx(th_j1)

    def test_from_samples_n_jobs_2(self):
        """Explicit n_jobs=2 should give same result as serial."""
        score = MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT)
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(50, 20, 1))
        th_serial = calibrate_threshold_from_samples(
            score=score, samples=samples, false_alarm_probability=0.05, parallel=False
        )
        th_j2 = calibrate_threshold_from_samples(
            score=score,
            samples=samples,
            false_alarm_probability=0.05,
            parallel=True,
            n_jobs=2,
        )
        assert th_serial == pytest.approx(th_j2)

    def test_from_data_n_jobs_2(self):
        score = MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT)
        data = np.random.default_rng(0).normal(size=100)
        th_serial = calibrate_threshold_from_data(
            score=score,
            training_data=data,
            false_alarm_probability=0.05,
            stream_len=30,
            n_paths=50,
            rng=42,
            parallel=False,
        )
        th_j2 = calibrate_threshold_from_data(
            score=score,
            training_data=data,
            false_alarm_probability=0.05,
            stream_len=30,
            n_paths=50,
            rng=42,
            parallel=True,
            n_jobs=2,
        )
        assert th_serial == pytest.approx(th_j2)
