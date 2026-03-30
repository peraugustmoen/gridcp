"""Extended tests targeting important uncovered code paths.

Test groups
-----------
1. get_G_grid correctness
   The core grid-building utility was never tested directly.

2. Score helper functions (as_obs, inv_sqrtm_pd)
   These are used inside every score implementation but had zero direct tests.

3. Penalty computation for every score type
   The penalty functions control false-alarm rates and were completely untested.
   We verify both TIME_DEPENDENT (grows with n) and CONSTANT (returns 1) modes.

4. Calibration internal helpers
   _normalize_rng, _resolve_changepoint, _infer_observation_mode,
   _validate_sampler_preflight, _resolve_n_jobs — all are safety-critical
   validation/setup code with no tests.

5. Detector edge cases
   Threshold validation, wrong-shape threshold errors, first-update output.

6. Score model edge cases
   Observation size mismatch, wrong-dimension errors.
"""

import numpy as np
import pytest
from gridcp.calibration import (
    _infer_observation_mode,
    _normalize_rng,
    _resolve_changepoint,
    _resolve_n_jobs,
    _validate_sampler_preflight,
)
from gridcp.detector import GridDetector
from gridcp.scores import (
    MeanCUSUM,
    MeanCUSUMUnknownVariance,
    MeanOrVariance,
    MultivariateMeanIdentityCov,
    MultivariateMeanOrCovariance,
    MultivariateMeanUnknownCov,
    RegressionDirect,
    RegressionMcScan,
    Variance,
)
from gridcp.scores._score_helpers import as_obs, inv_sqrtm_pd
from gridcp.typing import PenaltyType
from gridcp.utils import get_G_grid, get_changeloc_grid

# ---------------------------------------------------------------------------
# 1. get_G_grid correctness
# ---------------------------------------------------------------------------


class TestGetGGrid:
    """get_G_grid was never directly tested despite being the foundation of
    get_changeloc_grid.  These tests verify the grid definition:
      G^(t) always contains 1, all elements are in [1, t], the grid is sorted,
    and the relationship  changeloc_grid == t - reversed(G_grid)  holds.
    """

    def test_t1_returns_one(self):
        """G^(1) = {1} by definition."""
        assert np.array_equal(get_G_grid(1), np.array([1]))

    def test_t2(self):
        """G^(2) = {1} — at t=2 the grid has one element."""
        g = get_G_grid(2)
        assert np.array_equal(g, np.array([1]))

    def test_always_contains_one(self):
        """The grid must always contain 1 (the earliest candidate)."""
        for t in range(1, 50):
            assert 1 in get_G_grid(t), f"G^({t}) missing 1"

    def test_elements_in_range(self):
        """Every grid element must be in [1, t]."""
        for t in range(1, 50):
            g = get_G_grid(t)
            assert np.all(g >= 1) and np.all(g <= t), f"G^({t}) out of range"

    def test_sorted_ascending(self):
        """The grid is documented as sorted smallest-to-largest."""
        for t in range(1, 50):
            g = get_G_grid(t)
            assert np.all(g[:-1] <= g[1:]), f"G^({t}) not sorted"

    def test_consistency_with_changeloc_grid(self):
        """changeloc(t) == t - G(t)[::-1] — the defining relationship."""
        for t in range(1, 50):
            g = get_G_grid(t)
            expected_changeloc = t - g[::-1]
            actual_changeloc = get_changeloc_grid(t)
            assert np.array_equal(actual_changeloc, expected_changeloc), (
                f"Mismatch at t={t}"
            )

    def test_invalid_t_raises(self):
        """t < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            get_G_grid(0)
        with pytest.raises(ValueError):
            get_G_grid(-5)


# ---------------------------------------------------------------------------
# 2. Score helper functions
# ---------------------------------------------------------------------------


class TestAsObs:
    """as_obs normalises observations to 1-D float64 vectors.
    It is used inside every score's update() but was never tested directly.
    We check: scalar input, list input, wrong-size error.
    """

    def test_scalar_to_1d(self):
        """A scalar observation with n_features=1 produces shape (1,)."""
        result = as_obs(3.5, n_features=1)
        assert result.shape == (1,)
        assert result.dtype == np.float64
        assert result[0] == 3.5

    def test_list_to_1d(self):
        """A list is flattened to a 1-D float64 array."""
        result = as_obs([1, 2, 3], n_features=3)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
        assert result.dtype == np.float64

    def test_2d_array_flattened(self):
        """A 2-D array with correct total size is reshaped to 1-D."""
        result = as_obs(np.array([[1.0, 2.0]]), n_features=2)
        assert result.shape == (2,)

    def test_wrong_size_raises(self):
        """Size mismatch raises ValueError with informative message."""
        with pytest.raises(ValueError, match="Expected observation of size 3"):
            as_obs([1.0, 2.0], n_features=3)


class TestInvSqrtmPd:
    """inv_sqrtm_pd computes the inverse matrix square root of a positive-
    definite matrix.  It is used in RegressionDirect but was never tested.
    We verify: identity → identity, known 2×2 case, A^{-1/2} A A^{-1/2} = I.
    """

    def test_identity_returns_identity(self):
        """inv_sqrtm(I) = I."""
        identity = np.eye(3)
        result = inv_sqrtm_pd(identity)
        np.testing.assert_allclose(result, identity, atol=1e-12)

    def test_diagonal_matrix(self):
        """inv_sqrtm(diag(d)) = diag(1/sqrt(d))."""
        d = np.array([4.0, 9.0, 16.0])
        A = np.diag(d)
        result = inv_sqrtm_pd(A)
        expected = np.diag(1.0 / np.sqrt(d))
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_product_gives_identity(self):
        """A^{-1/2} A A^{-1/2} = I for a random PD matrix."""
        rng = np.random.default_rng(42)
        L = rng.standard_normal((4, 4))
        A = L @ L.T + np.eye(4)  # guaranteed PD
        A_inv_sqrt = inv_sqrtm_pd(A)
        product = A_inv_sqrt @ A @ A_inv_sqrt
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    def test_symmetric_output(self):
        """Output must be symmetric."""
        rng = np.random.default_rng(7)
        L = rng.standard_normal((3, 3))
        A = L @ L.T + np.eye(3)
        result = inv_sqrtm_pd(A)
        np.testing.assert_allclose(result, result.T, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Penalty computation
# ---------------------------------------------------------------------------


class TestPenalties:
    """Every score model's _get_penalty() was completely untested.
    The penalty controls the false-alarm rate: TIME_DEPENDENT should grow with
    n_samples, and CONSTANT should always return 1.0.
    """

    SCORES_WITH_DEFAULTS = [
        MeanCUSUM(n_features=1),
        Variance(n_features=1),
        MeanOrVariance(n_features=1),
        MeanCUSUMUnknownVariance(n_features=1),
        MultivariateMeanIdentityCov(n_features=2),
        MultivariateMeanUnknownCov(n_features=2),
        MultivariateMeanOrCovariance(n_features=2),
        RegressionDirect(n_regressors=2),
        RegressionMcScan(n_regressors=2),
    ]

    @pytest.mark.parametrize(
        "score", SCORES_WITH_DEFAULTS, ids=lambda s: type(s).__name__
    )
    def test_time_dependent_penalty_increases(self, score):
        """TIME_DEPENDENT penalty must be strictly increasing in n_samples."""
        p10 = np.asarray(score._get_penalty(10))
        p100 = np.asarray(score._get_penalty(100))
        p1000 = np.asarray(score._get_penalty(1000))
        assert np.all(p10 > 0)
        assert np.all(p100 > p10)
        assert np.all(p1000 > p100)

    @pytest.mark.parametrize(
        "score_cls",
        [
            MeanCUSUM,
            Variance,
            MeanOrVariance,
            MeanCUSUMUnknownVariance,
            MultivariateMeanIdentityCov,
            MultivariateMeanUnknownCov,
            MultivariateMeanOrCovariance,
        ],
        ids=lambda c: c.__name__,
    )
    def test_constant_penalty_is_one(self, score_cls):
        """CONSTANT penalty must return 1.0 regardless of n_samples."""
        score = score_cls(n_features=2, penalty=PenaltyType.CONSTANT)
        np.testing.assert_array_equal(np.asarray(score._get_penalty(10)), 1.0)
        np.testing.assert_array_equal(np.asarray(score._get_penalty(1000)), 1.0)


# ---------------------------------------------------------------------------
# 4. Calibration internal helpers
# ---------------------------------------------------------------------------


class TestNormalizeRng:
    """_normalize_rng converts int/None/Generator → Generator.
    It is the entry point for all MC randomness and was completely untested.
    """

    def test_none_gives_deterministic_generator(self):
        """None → fixed-seed generator; two calls produce same sequence."""
        g1 = _normalize_rng(None)
        g2 = _normalize_rng(None)
        assert g1.random() == g2.random()

    def test_int_seed(self):
        """An integer seed produces a reproducible generator."""
        g1 = _normalize_rng(42)
        g2 = _normalize_rng(42)
        assert g1.random() == g2.random()

    def test_generator_passthrough(self):
        """An existing Generator is returned as-is."""
        g = np.random.default_rng(7)
        assert _normalize_rng(g) is g

    def test_invalid_type_raises(self):
        """Non-int, non-None, non-Generator raises TypeError."""
        with pytest.raises(TypeError, match="rng must be"):
            _normalize_rng("hello")


class TestResolveChangepoint:
    """_resolve_changepoint resolves None / int / callable → int.
    It validates boundaries and was completely untested.
    """

    def test_none_returns_n_samples(self):
        """None means "no changepoint" → returns stream length."""
        rng = np.random.default_rng(0)
        assert _resolve_changepoint(None, rng, 100, 0) == 100

    def test_int_passthrough(self):
        """An integer in [0, n_samples] is returned as-is."""
        rng = np.random.default_rng(0)
        assert _resolve_changepoint(50, rng, 100, 0) == 50

    def test_boundary_zero(self):
        """changepoint=0 (all post-change) is valid."""
        rng = np.random.default_rng(0)
        assert _resolve_changepoint(0, rng, 100, 0) == 0

    def test_boundary_n_samples(self):
        """changepoint=n_samples (all pre-change) is valid."""
        rng = np.random.default_rng(0)
        assert _resolve_changepoint(100, rng, 100, 0) == 100

    def test_out_of_range_raises(self):
        """changepoint outside [0, n_samples] raises ValueError."""
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="changepoint must be in"):
            _resolve_changepoint(101, rng, 100, 0)
        with pytest.raises(ValueError, match="changepoint must be in"):
            _resolve_changepoint(-1, rng, 100, 0)

    def test_callable(self):
        """A callable is invoked with (rng, n_samples, path_index)."""
        rng = np.random.default_rng(0)
        result = _resolve_changepoint(lambda r, n, p: n // 2, rng, 100, 0)
        assert result == 50


class TestInferObservationMode:
    """_infer_observation_mode classifies sampler output as scalar or vector.
    This affects how observations are normalised in the MC hot loop.
    """

    def test_scalar_float(self):
        assert _infer_observation_mode(3.14, n_features=1) == "scalar"

    def test_scalar_int(self):
        assert _infer_observation_mode(5, n_features=1) == "scalar"

    def test_vector(self):
        assert _infer_observation_mode(np.array([1.0, 2.0]), n_features=2) == "vector"

    def test_wrong_size_vector_raises(self):
        """Vector with wrong size raises ValueError."""
        with pytest.raises(ValueError, match="wrong size"):
            _infer_observation_mode(np.array([1.0, 2.0, 3.0]), n_features=2)


class TestValidateSamplerPreflight:
    """_validate_sampler_preflight probes one sample from a sampler to check
    compatibility.  It catches shape/type mismatches before an expensive MC run.
    """

    def test_valid_scalar_sampler(self):
        """A scalar sampler with n_features=1 passes."""
        _validate_sampler_preflight(
            lambda rng: rng.normal(),
            sampler_name="test",
            n_features=1,
            args=(),
            kwargs={},
        )

    def test_valid_vector_sampler(self):
        """A vector sampler returning correct size passes."""
        _validate_sampler_preflight(
            lambda rng: rng.normal(size=3),
            sampler_name="test",
            n_features=3,
            args=(),
            kwargs={},
        )

    def test_wrong_size_raises(self):
        """Sampler returning wrong vector size raises ValueError."""
        with pytest.raises(ValueError, match="total size"):
            _validate_sampler_preflight(
                lambda rng: rng.normal(size=5),
                sampler_name="bad_sampler",
                n_features=3,
                args=(),
                kwargs={},
            )

    def test_scalar_for_multivariate_raises(self):
        """A scalar sampler with n_features > 1 raises ValueError."""
        with pytest.raises(ValueError, match="scalar"):
            _validate_sampler_preflight(
                lambda rng: rng.normal(),
                sampler_name="test",
                n_features=3,
                args=(),
                kwargs={},
            )


class TestResolveNJobs:
    """_resolve_n_jobs clamps the worker count to [1, n_paths]."""

    def test_explicit_value(self):
        assert _resolve_n_jobs(2, n_paths=10) == 2

    def test_clamped_to_n_paths(self):
        """n_jobs > n_paths is clamped down."""
        assert _resolve_n_jobs(100, n_paths=3) == 3

    def test_none_uses_cpu_count(self):
        """None falls back to os.cpu_count()."""
        result = _resolve_n_jobs(None, n_paths=1000)
        assert 1 <= result <= 1000

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="n_jobs must be >= 1"):
            _resolve_n_jobs(0, n_paths=10)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="n_jobs must be >= 1"):
            _resolve_n_jobs(-1, n_paths=10)


# ---------------------------------------------------------------------------
# 5. Detector edge cases
# ---------------------------------------------------------------------------


class TestDetectorEdgeCases:
    """GridDetector validation and threshold handling had partial coverage.
    These tests fill in: negative threshold, vector threshold shape mismatch,
    first-update output (n_samples < 2), and the full update-grid mechanism.
    """

    def test_negative_scalar_threshold_rejected(self):
        """Negative threshold must raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            GridDetector(score=MeanCUSUM(), threshold=-1.0)

    def test_zero_scalar_threshold_rejected(self):
        """Zero threshold must raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            GridDetector(score=MeanCUSUM(), threshold=0.0)

    def test_vector_threshold_negative_entry_rejected(self):
        """Vector threshold with any non-positive entry must raise."""
        with pytest.raises(ValueError, match="positive"):
            GridDetector(
                score=MultivariateMeanIdentityCov(n_features=2),
                threshold=np.array([1.0, -0.5]),
            )

    def test_2d_threshold_rejected(self):
        """2-D threshold array must raise ValueError."""
        with pytest.raises(ValueError, match="scalar or 1-D"):
            GridDetector(
                score=MeanCUSUM(),
                threshold=np.array([[1.0, 2.0]]),
            )

    def test_first_update_no_alarm(self):
        """The first observation (n_samples=1) must never alarm."""
        det = GridDetector(score=MeanCUSUM(), threshold=1.0)
        state = det.init_state()
        state, out = det.update(state, 999999.0)
        assert out["n_samples"] == 1
        assert not out["alarm"]
        assert out["max_score"] == 0.0

    def test_grid_matches_changeloc_grid(self):
        """After n updates, the detector grid must match get_changeloc_grid(n)."""
        n = 30
        det = GridDetector(score=MeanCUSUM(), threshold=1e10)
        state = det.init_state()
        rng = np.random.default_rng(0)
        for _ in range(n):
            state, _ = det.update(state, rng.normal())
        expected_grid = get_changeloc_grid(n)
        actual_grid = np.array(state.grid, dtype=np.int64)
        np.testing.assert_array_equal(actual_grid, expected_grid)


# ---------------------------------------------------------------------------
# 6. Score model observation-mismatch errors
# ---------------------------------------------------------------------------
class TestScoreObservationMismatch:
    """Each score must raise ValueError when given an observation with the
    wrong number of features.  These errors protect against silent bugs where
    e.g. a univariate sampler is paired with a multivariate score.
    """

    def test_mean_cusum_wrong_size(self):
        score = MeanCUSUM(n_features=3)
        state = score.init_state()
        with pytest.raises(ValueError, match="size"):
            score.update(state, [1.0, 2.0])

    def test_variance_wrong_size(self):
        score = Variance(n_features=2)
        state = score.init_state()
        with pytest.raises(ValueError, match="size"):
            score.update(state, [1.0, 2.0, 3.0])

    def test_regression_direct_wrong_size(self):
        score = RegressionDirect(n_regressors=2)  # n_features = 3
        state = score.init_state()
        with pytest.raises(ValueError, match="size"):
            score.update(state, [1.0])

    def test_multivariate_identity_cov_wrong_size(self):
        score = MultivariateMeanIdentityCov(n_features=4)
        state = score.init_state()
        with pytest.raises(ValueError, match="size"):
            score.update(state, [1.0, 2.0])
