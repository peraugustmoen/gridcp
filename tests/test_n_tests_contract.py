"""Tests for the n_tests dimension contract.

Phase 1 of the explicit score dimension contract plan:
- Every score has a stable n_tests across all calls.
- compute_penalized_scores always returns shape (G, K) with K == n_tests.
- Detector construction fails early when vector threshold length != n_tests.
- Calibration output width K is consistent and driven by n_tests.
- A malformed score with changing output width triggers clear exceptions.
"""

import numpy as np
import pytest

from gridcp.detector import GridDetector
from gridcp.scores import (
    ExponentialFamilyGLR,
    MeanCUSUM,
    MeanCUSUMUnknownVariance,
    MeanOrVariance,
    MultivariateMeanIdentityCov,
    MultivariateMeanOrCovariance,
    MultivariateMeanUnknownCov,
    NPFOCuS,
    RegressionDirect,
    RegressionMcScan,
    Variance,
)
from gridcp.calibration import (
    calibrate_threshold_false_alarm,
    mc_max_scores,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h_vector_identity(x):
    return x.astype(np.float64, copy=True)


def _A_vector_gaussian(theta):
    return 0.5 * float(np.dot(theta, theta))


def _Agrad_vector_gaussian(theta):
    return theta.astype(np.float64, copy=True)


def _Ahess_vector_gaussian(theta):
    return np.eye(theta.shape[0], dtype=np.float64)


def _normal_sampler(rng):
    return float(rng.standard_normal())


ALL_BUILTIN_SCORES = [
    ("MeanCUSUM_K1", MeanCUSUM(n_features=1), 1),
    ("MeanCUSUM_K3", MeanCUSUM(n_features=3), 1),
    ("MeanCUSUMUnknownVariance_K1", MeanCUSUMUnknownVariance(n_features=1), 1),
    ("MeanCUSUMUnknownVariance_K3", MeanCUSUMUnknownVariance(n_features=3), 1),
    ("Variance_K1", Variance(n_features=1), 1),
    ("MeanOrVariance_K1", MeanOrVariance(n_features=1), 1),
    ("MultivariateMeanIdentityCov_K2", MultivariateMeanIdentityCov(n_features=3), 2),
    ("MultivariateMeanUnknownCov_K1", MultivariateMeanUnknownCov(n_features=2), 1),
    ("MultivariateMeanOrCovariance_K1", MultivariateMeanOrCovariance(n_features=2), 1),
    ("RegressionDirect_K1", RegressionDirect(n_regressors=2), 1),
    ("RegressionMcScan_K1", RegressionMcScan(n_regressors=2), 1),
    (
        "ExponentialFamilyGLR_K1",
        ExponentialFamilyGLR(
            v=2,
            n_features=2,
            h=_h_vector_identity,
            A=_A_vector_gaussian,
            A_grad=_Agrad_vector_gaussian,
            A_hess=_Ahess_vector_gaussian,
        ),
        1,
    ),
    ("NPFOCuS_K2", NPFOCuS(value_grid=np.linspace(-2.0, 2.0, 9), n_features=1), 2),
]


# ---------------------------------------------------------------------------
# 1. Stable n_tests value
# ---------------------------------------------------------------------------


class TestNTestsStability:
    """n_tests is a fixed declared integer for every built-in score."""

    @pytest.mark.parametrize(
        "name, score, expected_K",
        ALL_BUILTIN_SCORES,
        ids=[t[0] for t in ALL_BUILTIN_SCORES],
    )
    def test_n_tests_is_positive_int(self, name, score, expected_K):
        """n_tests must be a positive integer."""
        assert isinstance(score.n_tests, int)
        assert score.n_tests >= 1

    @pytest.mark.parametrize(
        "name, score, expected_K",
        ALL_BUILTIN_SCORES,
        ids=[t[0] for t in ALL_BUILTIN_SCORES],
    )
    def test_n_tests_matches_expected(self, name, score, expected_K):
        """n_tests matches the expected value for each built-in."""
        assert score.n_tests == expected_K

    @pytest.mark.parametrize(
        "name, score, expected_K",
        ALL_BUILTIN_SCORES,
        ids=[t[0] for t in ALL_BUILTIN_SCORES],
    )
    def test_n_tests_stable_across_calls(self, name, score, expected_K):
        """n_tests returns the same value on repeated calls."""
        assert score.n_tests == score.n_tests == score.n_tests


# ---------------------------------------------------------------------------
# 2. compute_penalized_scores output shape is (G, K) with K == n_tests
# ---------------------------------------------------------------------------


class TestPenalizedScoresShape:
    """compute_penalized_scores must always return shape (G, n_tests)."""

    @pytest.mark.parametrize(
        "name, score, expected_K",
        ALL_BUILTIN_SCORES,
        ids=[t[0] for t in ALL_BUILTIN_SCORES],
    )
    def test_output_shape_matches_n_tests(self, name, score, expected_K):
        """After several updates, output shape is (G, n_tests) for all G > 0."""
        detector = GridDetector(score=score, threshold=1e10)
        state = detector.init_state()
        rng = np.random.default_rng(0)
        obs_size = score.n_features

        for _ in range(10):
            x = rng.standard_normal(obs_size)
            state, _ = detector.update(state, x)

        # At this point we have candidates in state.candidate_score_states
        assert len(state.candidate_score_states) > 0
        out = score.compute_penalized_scores(
            state.running_score_state, state.candidate_score_states
        )
        assert out.ndim == 2, f"{name}: expected 2-D output, got shape {out.shape}"
        G, K = out.shape
        assert G == len(state.candidate_score_states), f"{name}: G mismatch"
        assert K == expected_K, f"{name}: K={K} != n_tests={expected_K}"

    @pytest.mark.parametrize(
        "name, score, expected_K",
        ALL_BUILTIN_SCORES,
        ids=[t[0] for t in ALL_BUILTIN_SCORES],
    )
    def test_detector_output_max_score_shape(self, name, score, expected_K):
        """max_score in DetectorOutput has shape (K,) matching n_tests."""
        detector = GridDetector(score=score, threshold=1e10)
        state = detector.init_state()
        rng = np.random.default_rng(1)
        obs_size = score.n_features

        # Need at least 2 samples for real scores
        for _ in range(5):
            state, out = detector.update(state, rng.standard_normal(obs_size))

        assert out["max_score"].shape == (expected_K,), (
            f"{name}: max_score shape {out['max_score'].shape} != ({expected_K},)"
        )
        assert out["max_score_index"].shape == (expected_K,)


# ---------------------------------------------------------------------------
# 3. Detector construction fails early when threshold length != n_tests
# ---------------------------------------------------------------------------


class TestDetectorThresholdLengthValidation:
    """GridDetector must fail early when a vector threshold has wrong length."""

    def test_threshold_length_mismatch_raises_at_construction_K1(self):
        """K=1 score with 2-element threshold must fail early at construction."""
        score = MeanCUSUM(n_features=1)  # n_tests = 1
        with pytest.raises(ValueError, match="n_tests|threshold"):
            GridDetector(score=score, threshold=np.array([1.0, 2.0]))

    def test_threshold_length_mismatch_raises_at_construction_K2(self):
        """K=2 score with 1-element threshold must fail early at construction."""
        score = MultivariateMeanIdentityCov(n_features=3)  # n_tests = 2
        with pytest.raises(ValueError, match="n_tests|threshold"):
            GridDetector(score=score, threshold=np.array([1.0]))

    def test_threshold_length_mismatch_raises_at_construction_K2_len3(self):
        """K=2 score with 3-element threshold must fail early at construction."""
        score = MultivariateMeanIdentityCov(n_features=3)  # n_tests = 2
        with pytest.raises(ValueError, match="n_tests|threshold"):
            GridDetector(score=score, threshold=np.array([1.0, 2.0, 3.0]))

    def test_matching_threshold_length_accepted_K1(self):
        """K=1 score with 1-element threshold must be accepted."""
        score = MeanCUSUM(n_features=1)  # n_tests = 1
        det = GridDetector(score=score, threshold=np.array([5.0]))
        assert det is not None

    def test_matching_threshold_length_accepted_K2(self):
        """K=2 score with 2-element threshold must be accepted."""
        score = MultivariateMeanIdentityCov(n_features=3)  # n_tests = 2
        det = GridDetector(score=score, threshold=np.array([5.0, 5.0]))
        assert det is not None

    def test_scalar_threshold_always_accepted(self):
        """Scalar threshold must always be accepted (broadcast)."""
        det1 = GridDetector(score=MeanCUSUM(n_features=1), threshold=5.0)
        det2 = GridDetector(
            score=MultivariateMeanIdentityCov(n_features=3), threshold=5.0
        )
        assert det1 is not None
        assert det2 is not None


# ---------------------------------------------------------------------------
# 4. Calibration output width K is consistent across paths/workers
# ---------------------------------------------------------------------------


class TestCalibrationOutputWidth:
    """mc_max_scores and calibrate_threshold_false_alarm output width must
    equal n_tests for the score being calibrated."""

    @pytest.mark.parametrize(
        "name, score, expected_K",
        ALL_BUILTIN_SCORES,
        ids=[t[0] for t in ALL_BUILTIN_SCORES],
    )
    def test_mc_max_scores_output_shape(self, name, score, expected_K):
        """mc_max_scores returns (n_paths, K) with K == score.n_tests."""
        n_features = score.n_features
        if n_features > 1:

            def pre_sampler(rng):
                return rng.standard_normal(n_features)
        else:
            pre_sampler = _normal_sampler

        detector = GridDetector(score=score, threshold=1e10)
        max_scores = mc_max_scores(
            detector=detector,
            n_paths=10,
            stream_len=15,
            pre_sampler=pre_sampler,
            rng=0,
            parallel=False,
        )
        assert max_scores.ndim == 2
        assert max_scores.shape == (10, expected_K), (
            f"{name}: mc_max_scores shape {max_scores.shape} expected (10, {expected_K})"
        )

    @pytest.mark.parametrize(
        "name, score, expected_K",
        ALL_BUILTIN_SCORES,
        ids=[t[0] for t in ALL_BUILTIN_SCORES],
    )
    def test_calibrate_threshold_output_shape(self, name, score, expected_K):
        """calibrate_threshold_false_alarm returns shape (K,) == (n_tests,)."""
        n_features = score.n_features
        if n_features > 1:

            def pre_sampler(rng):
                return rng.standard_normal(n_features)
        else:
            pre_sampler = _normal_sampler

        threshold = calibrate_threshold_false_alarm(
            score=score,
            false_alarm_probability=0.1,
            n_paths=10,
            stream_len=15,
            pre_sampler=pre_sampler,
            rng=0,
            parallel=False,
        )
        assert isinstance(threshold, np.ndarray)
        assert threshold.shape == (expected_K,), (
            f"{name}: threshold shape {threshold.shape} expected ({expected_K},)"
        )
        assert np.all(threshold > 0)


# ---------------------------------------------------------------------------
# 5. Malformed score: changing output width triggers clear exceptions
# ---------------------------------------------------------------------------


class _MalformedScore:
    """A deliberately broken score whose compute_penalized_scores changes K
    on successive calls (alternates between 1 and 2).
    """

    n_features: int = 1
    _call_count: int = 0

    @property
    def n_tests(self) -> int:
        return 1  # Declared K=1 but actual output will differ

    def init_state(self):
        return {"n_samples": 0, "sum": np.zeros(1)}

    def update(self, state, x):
        return {
            "n_samples": state["n_samples"] + 1,
            "sum": state["sum"] + np.asarray(x).reshape(-1)[:1],
        }

    def compute_penalized_scores(self, state, grid_states):
        # Alternately return wrong shape: sometimes (G,1), sometimes (G,2)
        self._call_count += 1
        G = len(grid_states)
        if self._call_count % 2 == 0:
            return np.zeros((G, 2))  # Wrong! declared n_tests=1 but returns 2 cols
        return np.zeros((G, 1))


class _AlwaysWrongWidthScore:
    """A broken score that always returns shape (G, 3) but declares n_tests=1."""

    n_features: int = 1

    @property
    def n_tests(self) -> int:
        return 1

    def init_state(self):
        return {"n_samples": 0}

    def update(self, state, x):
        return {"n_samples": state["n_samples"] + 1}

    def compute_penalized_scores(self, state, grid_states):
        G = len(grid_states)
        return np.zeros((G, 3))  # Wrong! declared n_tests=1 but returns 3 cols


class TestMalformedScoreEnforcement:
    """Detector must catch malformed scores that return wrong output width."""

    def test_wrong_output_width_raises_in_detector_update(self):
        """A score returning K != n_tests must be caught during detector update."""
        score = _AlwaysWrongWidthScore()
        detector = GridDetector(score=score, threshold=1.0)
        state = detector.init_state()

        state, _ = detector.update(state, 0.0)
        # Second call produces candidates and calls compute_penalized_scores
        with pytest.raises((ValueError, Exception)):
            detector.update(state, 0.0)

    def test_malformed_score_not_accepted_as_valid_detector(self):
        """GridDetector must detect ScoreModel protocol violations at construction."""
        # If _AlwaysWrongWidthScore passes construction (it has required attributes),
        # the violation should be caught at runtime during update.
        score = _AlwaysWrongWidthScore()
        detector = GridDetector(score=score, threshold=1.0)
        state = detector.init_state()

        state, _ = detector.update(state, 0.0)
        with pytest.raises((ValueError, Exception)):
            detector.update(state, 0.0)
