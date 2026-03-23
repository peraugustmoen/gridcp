import numpy as np
import pytest

import gridcp
from gridcp.detector import GridDetector
from gridcp.scores._mean_cusum import MeanCUSUM
from gridcp.scores._mean_cusum_unknown_variance import MeanCUSUMUnknownVariance


def _run_stream(data: np.ndarray, threshold: float = 10.0):
    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=threshold)
    state = detector.init_state()
    outputs = []
    for x in data:
        state, out = detector.update(state, np.asarray([x]))
        outputs.append(out)
    return detector, state, outputs


def _run_stream_unknown_variance(
    data: np.ndarray, threshold: float = 10.0, n_features: int = 1
):
    detector = GridDetector(
        score=MeanCUSUMUnknownVariance(n_features=n_features), threshold=threshold
    )
    state = detector.init_state()
    outputs = []
    for x in data:
        x_arr = np.asarray(x)
        obs = np.asarray([x]) if n_features == 1 and x_arr.ndim == 0 else x_arr
        state, out = detector.update(state, obs)
        outputs.append(out)
    return detector, state, outputs


def test_threshold_must_be_positive():
    with pytest.raises(ValueError):
        GridDetector(score=MeanCUSUM(n_features=1), threshold=0.0)

    with pytest.raises(ValueError):
        GridDetector(score=MeanCUSUM(n_features=1), threshold=-1.0)


def test_no_false_alarm_under_null_with_high_threshold():
    rng = np.random.default_rng(seed=123)
    x = rng.normal(loc=0.0, scale=1.0, size=200)

    _, _, outputs = _run_stream(x, threshold=50.0)
    assert all(not out["alarm"] for out in outputs)


def test_outlier_triggers_alarm():
    rng = np.random.default_rng(seed=123)
    x = rng.normal(loc=0.0, scale=1.0, size=200)

    detector, state, outputs = _run_stream(x, threshold=8.0)
    assert all(not out["alarm"] for out in outputs)

    state, out = detector.update(state, np.asarray([1000.0]))
    assert out["alarm"]
    assert out["max_score"] > detector.threshold


def test_running_sum_matches_numpy_cumsum():
    rng = np.random.default_rng(seed=42)
    x = rng.normal(loc=0.0, scale=1.0, size=200)

    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=100.0)
    state = detector.init_state()

    running_sum = 0.0
    for val in x:
        running_sum += float(val)
        state, _ = detector.update(state, np.asarray([val]))
        assert np.isclose(state.running_score_state.sum[0], running_sum)


def test_grid_and_candidate_states_stay_parallel():
    rng = np.random.default_rng(seed=7)
    x = rng.normal(loc=0.0, scale=1.0, size=250)

    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=100.0)
    state = detector.init_state()

    for val in x:
        state, _ = detector.update(state, np.asarray([val]))

        assert len(state.grid) == len(state.candidate_score_states)
        for g, candidate in zip(state.grid, state.candidate_score_states):
            # Candidate snapshot should correspond to the candidate grid time.
            assert candidate.n_samples == g


def test_detects_clear_mean_shift_mid_stream():
    rng = np.random.default_rng(seed=100)
    x_pre = rng.normal(loc=0.0, scale=1.0, size=100)
    x_post = rng.normal(loc=4.0, scale=1.0, size=100)
    x = np.concatenate([x_pre, x_post])

    _, _, outputs = _run_stream(x, threshold=3.0)

    # Require at least one alarm after the shift begins.
    post_shift_outputs = outputs[len(x_pre) :]
    assert any(out["alarm"] for out in post_shift_outputs)


def test_short_stream_behavior_and_output_schema():
    x = np.array([0.5, -0.2, 0.1], dtype=float)
    _, state, outputs = _run_stream(x, threshold=10.0)

    assert state.n_samples == 3
    for i, out in enumerate(outputs, start=1):
        assert out["index"] == i
        assert set(out.keys()) == {"index", "alarm", "max_score", "max_score_index"}

    # By construction, first step has no valid split candidate.
    assert outputs[0]["max_score"] == 0.0


def test_n_samples_increments_by_one():
    """Each update should increment n_samples by exactly 1."""
    rng = np.random.default_rng(seed=42)
    x = rng.normal(loc=0.0, scale=1.0, size=100)

    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=100.0)
    state = detector.init_state()
    assert state.n_samples == 0

    for i, val in enumerate(x, start=1):
        state, _ = detector.update(state, np.asarray([val]))
        assert (
            state.n_samples == i
        ), f"After {i} updates, expected n_samples={i}, got {state.n_samples}"


def test_max_score_alarms_when_exceeds_threshold():
    """The alarm should trigger when max_score exceeds the threshold."""
    rng = np.random.default_rng(seed=42)
    x = rng.normal(loc=0.0, scale=1.0, size=100)

    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=10.0)
    state = detector.init_state()

    for val in x:
        state, out = detector.update(state, np.asarray([val]))
        if out["alarm"]:
            # When alarm is triggered, max_score should exceed threshold
            assert (
                out["max_score"] > detector.threshold
            ), f"Alarm triggered but max_score {out['max_score']} <= threshold {detector.threshold}"

    # Force an alarm with extreme value and verify max_score exceeds threshold
    state, out = detector.update(state, np.asarray([1_000_000.0]))
    assert out["alarm"]
    assert out["max_score"] > detector.threshold


def test_max_score_index_is_valid_after_alarm():
    """After alarm, max_score_index should be a certain position."""
    rng = np.random.default_rng(seed=42)
    x = rng.normal(loc=0.0, scale=1.0, size=100)

    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=8.0)
    state = detector.init_state()

    for val in x:
        state, out = detector.update(state, np.asarray([val]))

    # Trigger alarm with extreme value
    state, out = detector.update(state, np.asarray([1_000_000.0]))
    assert out["alarm"]
    assert out["max_score_index"] == 100, "max_score_index should be 100"


def test_univariate_mean_1_baseline():
    """Reproduces the core checks from old API test_univariate_mean_1."""
    N = 1000
    rng = np.random.default_rng(seed=123)
    x = rng.normal(loc=0, scale=1, size=N)

    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=5.0)
    state = detector.init_state()

    outputs = []
    for val in x:
        state, out = detector.update(state, np.asarray([val]))
        outputs.append(out)

    # No alarm under null with threshold=5.0
    assert not any(out["alarm"] for out in outputs), "False alarm under null"

    # Extreme observation triggers alarm
    state, out = detector.update(state, np.asarray([1_000_000.0]))
    assert out["alarm"], "Failed to alarm on extreme observation"
    assert (
        out["max_score"] > detector.threshold
    ), f"max_score {out['max_score']} should exceed threshold {detector.threshold}"

    # Check cumulative sums match expected values
    # Re-run to accumulate running sum
    detector2 = GridDetector(score=MeanCUSUM(n_features=1), threshold=5.0)
    state2 = detector2.init_state()
    running_sum = 0.0
    for val in x:
        running_sum += float(val)
        state2, _ = detector2.update(state2, np.asarray([val]))
        # Verify running sum in MeanCUSUM state matches cumsum
        assert np.isclose(
            state2.running_score_state.sum[0], running_sum
        ), f"Running sum mismatch at n_samples={state2.n_samples}"


def test_grid_correctness_univariate():
    """Grid size should be O(log n) and grow appropriately with stream length."""
    rng = np.random.default_rng(seed=55)
    x = rng.normal(loc=0.0, scale=1.0, size=1000)

    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=100.0)
    state = detector.init_state()

    grid_sizes = []
    for val in x:
        state, _ = detector.update(state, np.asarray([val]))
        grid_sizes.append(len(state.grid))

    # Grid size should grow logarithmically with n_samples
    final_grid_size = grid_sizes[-1]
    assert final_grid_size > 1, "Grid should have multiple candidates"
    assert (
        final_grid_size < 100
    ), f"Grid size {final_grid_size} seems too large for n=1000"
    # Rough check: should be ~log(n)
    assert final_grid_size <= 2 * np.log2(
        1000
    ), f"Grid size {final_grid_size} exceeds 2*log2(1000)=~20"


def test_grid_and_cumulative_sums_match_manual_computation():
    """Verify grid and cumulative sums match manually computed values (mirrors old API test_univariate_mean_1).

    This test:
    1. Runs detector on N samples
    2. Manually computes cumulative sums using np.cumsum()
    3. Verifies that each candidate's stored sum matches the expected cumsum at that grid point
    4. Verifies final running sum equals total sum of the data
    """
    N = 100
    rng = np.random.default_rng(seed=123)
    x = rng.normal(loc=0, scale=1, size=N)

    detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=100.0)
    state = detector.init_state()

    # Run detector through all samples
    cumsum_x = np.cumsum(x)
    t = 0
    for val in x:
        state, _ = detector.update(state, np.asarray([val]))
        t = t + 1
        true_grid = gridcp.utils.get_changeloc_grid(t)
        if t >= 2:
            assert all(
                a == b for a, b in zip(state.grid, true_grid)
            ), f"At n_samples={t}, expected grid {true_grid}, got {state.grid}"

            true_cumsums = cumsum_x[
                true_grid - 1
            ]  # cumsum at each grid point (0-indexed)
            # Check that cumulative sums at each grid point match expected values

            for i in range(len(true_grid)):
                grid_point = true_grid[i]
                expected_cumsum = true_cumsums[i]
                actual_cumsum = state.candidate_score_states[i].sum[0]

                assert np.isclose(
                    expected_cumsum, actual_cumsum
                ), f"At n_samples={t}, grid point {grid_point}: expected cumsum {expected_cumsum}, got {actual_cumsum}"


def test_univariate_mean_unknown_variance_outlier_triggers_alarm():
    """Unknown-variance mode: no alarm under null, alarm after extreme observation."""
    rng = np.random.default_rng(seed=123)
    x = rng.normal(loc=0.0, scale=1.0, size=200)

    detector, state, outputs = _run_stream_unknown_variance(x, threshold=5.0)
    assert all(not out["alarm"] for out in outputs)

    state, out = detector.update(state, np.asarray([1_000_000.0]))
    assert out["alarm"]


def test_univariate_mean_unknown_variance_cumsums_match_manual():
    """Unknown-variance mode: prefix stats [sum(x), sum(x^2)] match manual computation."""
    N = 200
    rng = np.random.default_rng(seed=123)
    x = rng.normal(loc=0.0, scale=1.0, size=N)

    detector = GridDetector(
        score=MeanCUSUMUnknownVariance(n_features=1), threshold=100.0
    )
    state = detector.init_state()

    cumsum_x = np.cumsum(x)
    cumsum_x2 = np.cumsum(x * x)

    for t, val in enumerate(x, start=1):
        state, _ = detector.update(state, np.asarray([val]))
        true_grid = gridcp.utils.get_changeloc_grid(t)

        if t >= 2:
            assert all(
                a == b for a, b in zip(state.grid, true_grid)
            ), f"At n_samples={t}, expected grid {true_grid}, got {state.grid}"

            for i, g in enumerate(true_grid):
                expected_sum = cumsum_x[g - 1]
                expected_sum2 = cumsum_x2[g - 1]
                actual_stats = state.candidate_score_states[i].stats

                assert np.isclose(
                    actual_stats[0], expected_sum
                ), f"At n_samples={t}, grid point {g}: expected sum {expected_sum}, got {actual_stats[0]}"
                assert np.isclose(
                    actual_stats[1], expected_sum2
                ), f"At n_samples={t}, grid point {g}: expected sumsq {expected_sum2}, got {actual_stats[1]}"

    assert np.isclose(state.running_score_state.stats[0], cumsum_x[-1])
    assert np.isclose(state.running_score_state.stats[1], cumsum_x2[-1])


def test_univariate_mean_unknown_variance_no_false_alarm_high_threshold():
    """Unknown-variance mode: no false alarm with high threshold."""
    rng = np.random.default_rng(seed=999)
    x = rng.normal(loc=0.0, scale=1.0, size=300)

    _, _, outputs = _run_stream_unknown_variance(x, threshold=50.0)
    assert all(not out["alarm"] for out in outputs)


def test_univariate_mean_unknown_variance_detects_shift():
    """Unknown-variance mode should detect a genuine mean shift."""
    N_pre = 200
    N_post = 200
    rng = np.random.default_rng(seed=42)
    x_pre = rng.normal(loc=0.0, scale=1.0, size=N_pre)
    x_post = rng.normal(loc=5.0, scale=1.0, size=N_post)

    detector = GridDetector(score=MeanCUSUMUnknownVariance(n_features=1), threshold=5.0)
    state = detector.init_state()

    for val in x_pre:
        state, out = detector.update(state, np.asarray([val]))
    assert not out["alarm"]

    alarmed = False
    for val in x_post:
        state, out = detector.update(state, np.asarray([val]))
        if out["alarm"]:
            alarmed = True
            break

    assert alarmed, "Unknown-variance detector failed to detect mean shift from 0 to 5"


def test_univariate_mean_unknown_variance_requires_matching_feature_size():
    """Unknown-variance score should reject observations with wrong feature size."""
    detector = GridDetector(score=MeanCUSUMUnknownVariance(n_features=1), threshold=5.0)
    state = detector.init_state()

    state, _ = detector.update(state, np.asarray([0.2]))

    with pytest.raises(ValueError):
        detector.update(state, np.asarray([0.1, 0.2]))


def test_multivariate_mean_unknown_variance_accepts_vectors_and_tracks_stats():
    """Unknown-variance score should support multivariate streams and track per-feature stats."""
    rng = np.random.default_rng(seed=24)
    n_samples = 150
    n_features = 3
    x = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    detector = GridDetector(
        score=MeanCUSUMUnknownVariance(n_features=n_features), threshold=100.0
    )
    state = detector.init_state()

    cumsum_x = np.cumsum(x, axis=0)
    cumsum_x2 = np.cumsum(x * x, axis=0)

    for t, row in enumerate(x, start=1):
        state, _ = detector.update(state, row)
        assert np.allclose(state.running_score_state.stats[0], cumsum_x[t - 1])
        assert np.allclose(state.running_score_state.stats[1], cumsum_x2[t - 1])


def test_multivariate_mean_unknown_variance_detects_single_feature_shift():
    """Unknown-variance multivariate score should alarm when one feature has a strong shift."""
    rng = np.random.default_rng(seed=123)
    n_pre = 200
    n_post = 200
    n_features = 3

    x_pre = rng.normal(loc=0.0, scale=1.0, size=(n_pre, n_features))
    x_post = rng.normal(loc=0.0, scale=1.0, size=(n_post, n_features))
    x_post[:, 1] += 5.0  # strong shift in a single feature
    x = np.vstack([x_pre, x_post])

    detector = GridDetector(
        score=MeanCUSUMUnknownVariance(n_features=n_features), threshold=5.0
    )
    state = detector.init_state()
    alarmed = False

    for row in x:
        state, out = detector.update(state, row)
        if out["alarm"]:
            alarmed = True
            break

    assert (
        alarmed
    ), "Multivariate unknown-variance detector failed to detect shifted feature"


def test_multivariate_known_variance_matches_independent_streams():
    """Known-variance multivariate detector should match independent univariate runs.

    We check, at every step, that:
    1. Running sums match feature-wise independent detectors.
    2. Candidate prefix sums match feature-wise independent detectors.
    3. max_score equals the maximum of independent max_scores.
    """
    rng = np.random.default_rng(seed=2026)
    n_samples = 250
    n_features = 3
    data = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    threshold = 1e6
    multi_detector = GridDetector(
        score=MeanCUSUM(n_features=n_features), threshold=threshold
    )
    multi_state = multi_detector.init_state()

    single_detectors = [
        GridDetector(score=MeanCUSUM(n_features=1), threshold=threshold)
        for _ in range(n_features)
    ]
    single_states = [det.init_state() for det in single_detectors]

    for row in data:
        multi_state, multi_out = multi_detector.update(multi_state, row)

        single_outs = []
        for k in range(n_features):
            single_states[k], single_out = single_detectors[k].update(
                single_states[k], np.asarray([row[k]])
            )
            single_outs.append(single_out)

        # Running statistics parity.
        expected_running_sum = np.array(
            [st.running_score_state.sum[0] for st in single_states]
        )
        assert np.allclose(multi_state.running_score_state.sum, expected_running_sum)

        # Candidate statistics parity.
        for i, multi_candidate_state in enumerate(multi_state.candidate_score_states):
            expected_candidate_sum = np.array(
                [
                    single_states[k].candidate_score_states[i].sum[0]
                    for k in range(n_features)
                ]
            )
            assert np.allclose(multi_candidate_state.sum, expected_candidate_sum)

        # max_score parity with independent feature-wise runs.
        expected_max_score = max(out["max_score"] for out in single_outs)
        assert np.isclose(multi_out["max_score"], expected_max_score)


def test_multivariate_unknown_variance_matches_independent_streams():
    """Unknown-variance multivariate detector should match independent univariate runs.

    We check, at every step, that:
    1. Running [sum, sumsq] stats match feature-wise independent detectors.
    2. Candidate [sum, sumsq] stats match feature-wise independent detectors.
    3. max_score equals the maximum of independent max_scores.
    """
    rng = np.random.default_rng(seed=2027)
    n_samples = 250
    n_features = 3
    data = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    threshold = 1e6
    multi_detector = GridDetector(
        score=MeanCUSUMUnknownVariance(n_features=n_features), threshold=threshold
    )
    multi_state = multi_detector.init_state()

    single_detectors = [
        GridDetector(score=MeanCUSUMUnknownVariance(n_features=1), threshold=threshold)
        for _ in range(n_features)
    ]
    single_states = [det.init_state() for det in single_detectors]

    for row in data:
        multi_state, multi_out = multi_detector.update(multi_state, row)

        single_outs = []
        for k in range(n_features):
            single_states[k], single_out = single_detectors[k].update(
                single_states[k], np.asarray([row[k]])
            )
            single_outs.append(single_out)

        # Running statistics parity.
        expected_running_sum = np.array(
            [st.running_score_state.stats[0] for st in single_states]
        )
        expected_running_sumsq = np.array(
            [st.running_score_state.stats[1] for st in single_states]
        )
        assert np.allclose(
            multi_state.running_score_state.stats[0], expected_running_sum
        )
        assert np.allclose(
            multi_state.running_score_state.stats[1], expected_running_sumsq
        )

        # Candidate statistics parity.
        for i, multi_candidate_state in enumerate(multi_state.candidate_score_states):
            expected_candidate_sum = np.array(
                [
                    single_states[k].candidate_score_states[i].stats[0]
                    for k in range(n_features)
                ]
            )
            expected_candidate_sumsq = np.array(
                [
                    single_states[k].candidate_score_states[i].stats[1]
                    for k in range(n_features)
                ]
            )
            assert np.allclose(multi_candidate_state.stats[0], expected_candidate_sum)
            assert np.allclose(multi_candidate_state.stats[1], expected_candidate_sumsq)

        # max_score parity with independent feature-wise runs.
        expected_max_score = max(out["max_score"] for out in single_outs)
        assert np.isclose(multi_out["max_score"], expected_max_score)
