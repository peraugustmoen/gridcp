"""Frozen reference copies of the legacy score kernels.

These are verbatim snapshots of the pre-redesign numba kernels (from the
``gridcp/scores/_mean_cusum.py``, ``_multivariate_mean_identity_cov.py``,
``_mean_unknown_variance.py``, ``_variance.py`` and ``_mean_or_variance.py``
modules) that were merged/renamed in the ``gridcp-score-redesign`` change.

They exist only so the reproduction tests can assert that the new scores match
the old behavior up to a small numerical tolerance.  Do not import them from
package code.
"""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def reference_mean_cusum_score(
    total_sum: np.ndarray,
    before_sums: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> np.ndarray:
    """Legacy ``mean_cusum_score``: returns ``max_j C_j - 1`` per candidate."""
    n_candidates = before_sums.shape[0]
    n_features = before_sums.shape[1]
    out = np.empty(n_candidates, dtype=np.float64)
    t = total_samples

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1
        before_weight = np.sqrt(n2 / (t * n1))
        after_weight = np.sqrt(n1 / (t * n2))

        s1 = before_sums[i, 0]
        s2 = total_sum[0] - s1
        row_max = (before_weight * s1 - after_weight * s2) ** 2
        for j in range(1, n_features):
            s1 = before_sums[i, j]
            s2 = total_sum[j] - s1
            sq = (before_weight * s1 - after_weight * s2) ** 2
            if sq > row_max:
                row_max = sq
        out[i] = row_max - 1.0

    return out


@nb.njit(cache=True)
def reference_mean_cusum_penalty(n_samples: int, n_features: int) -> float:
    """Legacy CUSUM penalty divisor ``log(t p) + sqrt(log(t p))``."""
    logg = np.log(n_samples * n_features)
    return logg + np.sqrt(logg)


@nb.njit(cache=True)
def reference_multivariate_mean_identity_cov_score(
    total_sum: np.ndarray,
    before_sums: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> np.ndarray:
    """Legacy identity-cov score: ``(G, 2)`` of ``[max - 1, sum - p]``."""
    n_candidates = before_sums.shape[0]
    n_features = before_sums.shape[1]
    out = np.zeros((n_candidates, 2), dtype=np.float64)
    t = total_samples

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1

        if n1 == 0 or n2 == 0:
            out[i, 0] = 0.0
            out[i, 1] = 0.0
            continue

        before_weight = np.sqrt(n2 / (t * n1))
        after_weight = np.sqrt(n1 / (t * n2))

        max_val = 0.0
        sum_val = 0.0
        for j in range(n_features):
            s1_j = before_sums[i, j]
            s2_j = total_sum[j] - s1_j
            sq = (before_weight * s1_j - after_weight * s2_j) ** 2
            sum_val += sq
            if sq > max_val:
                max_val = sq

        out[i, 0] = max_val - 1.0
        out[i, 1] = sum_val - n_features

    return out


@nb.njit(cache=True)
def reference_multivariate_mean_identity_cov_penalty(
    n_samples: int, n_features: int
) -> np.ndarray:
    """Legacy identity-cov penalty: ``[log(t)+log(p), sqrt(p log t)+log t]``."""
    t = n_samples
    p = n_features
    return np.array(
        [
            np.log(t) + np.log(p),
            np.sqrt(p * np.log(t)) + np.log(t),
        ]
    )


@nb.njit(cache=True)
def reference_mean_unknown_variance_score(
    total_stats: np.ndarray,
    before_stats: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> np.ndarray:
    """Legacy ``mean_unknown_variance_score``: max per-feature centered LR."""
    n_candidates = before_samples.shape[0]
    out = np.zeros(n_candidates, dtype=np.float64)
    n_features = total_stats.shape[1]

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = total_samples - n1

        if n1 <= 2 or n2 <= 2:
            out[i] = 0.0
            continue

        best_feature_score = -1.0e300
        has_valid_feature = False

        for k in range(n_features):
            total_sum = total_stats[0, k]
            total_sum2 = total_stats[1, k]
            sum1 = before_stats[i, 0, k]
            sum2_1 = before_stats[i, 1, k]

            sum2 = total_sum - sum1
            sum2_2 = total_sum2 - sum2_1

            sigma_null = (
                total_sum2 - total_sum * total_sum / total_samples
            ) / total_samples
            sigma_alt = (
                (sum2_1 - sum1 * sum1 / n1) + (sum2_2 - sum2 * sum2 / n2)
            ) / total_samples

            if sigma_null <= 0.0 or sigma_alt <= 0.0:
                continue

            has_valid_feature = True
            feature_lr = total_samples * np.log(sigma_null) - total_samples * np.log(
                sigma_alt
            )
            feature_score = feature_lr - 1.0
            if feature_score > best_feature_score:
                best_feature_score = feature_score

        out[i] = best_feature_score if has_valid_feature else 0.0

    return out


@nb.njit(cache=True)
def reference_variance_score(
    total_sum_sq: np.ndarray,
    before_sum_sqs: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> np.ndarray:
    """Legacy ``variance_score``: max per-feature centered LR."""
    n_candidates = before_sum_sqs.shape[0]
    n_features = before_sum_sqs.shape[1]
    out = np.zeros(n_candidates, dtype=np.float64)
    t = total_samples

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1

        if n1 == 0 or n2 == 0:
            out[i] = 0.0
            continue

        best = -1.0e300
        has_valid = False
        for k in range(n_features):
            sigma_tot = total_sum_sq[k] / t
            sigma_1 = before_sum_sqs[i, k] / n1
            sigma_2 = (total_sum_sq[k] - before_sum_sqs[i, k]) / n2
            if sigma_tot <= 0.0 or sigma_1 <= 0.0 or sigma_2 <= 0.0:
                continue
            has_valid = True
            lr = t * np.log(sigma_tot) - n1 * np.log(sigma_1) - n2 * np.log(sigma_2)
            score = lr - 1.0
            if score > best:
                best = score

        out[i] = best if has_valid else 0.0

    return out


@nb.njit(cache=True)
def reference_mean_or_variance_score(
    total_sum: np.ndarray,
    total_sum_sq: np.ndarray,
    before_sums: np.ndarray,
    before_sum_sqs: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> np.ndarray:
    """Legacy ``mean_or_variance_score``: max per-feature centered LR (df=2)."""
    n_candidates = before_sums.shape[0]
    n_features = before_sums.shape[1]
    out = np.zeros(n_candidates, dtype=np.float64)
    t = total_samples

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1

        if n1 <= 2 or n2 <= 2:
            out[i] = 0.0
            continue

        best = -1.0e300
        has_valid = False
        for k in range(n_features):
            s_tot = total_sum[k]
            s1 = before_sums[i, k]
            s2 = s_tot - s1
            ss_tot = total_sum_sq[k]
            ss1 = before_sum_sqs[i, k]
            ss2 = ss_tot - ss1

            sigma_tot = (ss_tot - s_tot * s_tot / t) / t
            sigma_1 = (ss1 - s1 * s1 / n1) / n1
            sigma_2 = (ss2 - s2 * s2 / n2) / n2
            if sigma_tot <= 0.0 or sigma_1 <= 0.0 or sigma_2 <= 0.0:
                continue

            has_valid = True
            lr = t * np.log(sigma_tot) - n1 * np.log(sigma_1) - n2 * np.log(sigma_2)
            score = lr - 2.0
            if score > best:
                best = score

        out[i] = best if has_valid else 0.0

    return out


@nb.njit(cache=True)
def reference_multivariate_mean_unknown_cov_score(
    total_sum: np.ndarray,
    total_sum_outer: np.ndarray,
    before_sums: np.ndarray,
    before_sum_outers: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Legacy full-covariance score: ``2 * LR - p`` per candidate."""
    n_candidates = before_sums.shape[0]
    out = np.zeros(n_candidates, dtype=np.float64)
    t = total_samples
    p = n_features
    df = float(p)

    if t < 2 * p + 2:
        return out

    s_tot = total_sum
    sxx_tot = total_sum_outer

    sigma_null = (sxx_tot - np.outer(s_tot, s_tot) / t) / t
    sign_null, logdet_null = np.linalg.slogdet(sigma_null)
    if sign_null <= 0:
        return out

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1

        s1 = before_sums[i]
        sxx1 = before_sum_outers[i]
        s2 = s_tot - s1
        sxx2 = sxx_tot - sxx1

        sigma_alt = (
            (sxx1 - np.outer(s1, s1) / n1) + (sxx2 - np.outer(s2, s2) / n2)
        ) / t
        sign1, logdet1 = np.linalg.slogdet(sigma_alt)
        if sign1 <= 0:
            out[i] = 0.0
            continue

        out[i] = t * (logdet_null - logdet1) - df

    return out


@nb.njit(cache=True)
def reference_multivariate_mean_unknown_cov_penalty(
    n_samples: int, n_features: int
) -> float:
    """Legacy full-covariance penalty ``sqrt(p log t) + log t``."""
    df = float(n_features)
    return np.sqrt(df * np.log(n_samples)) + np.log(n_samples)


@nb.njit(cache=True)
def reference_multivariate_mean_or_covariance_score(
    total_sum: np.ndarray,
    total_sum_outer: np.ndarray,
    before_sums: np.ndarray,
    before_sum_outers: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Legacy mean-or-covariance score: ``2 * LR - df`` per candidate."""
    n_candidates = before_sums.shape[0]
    out = np.zeros(n_candidates, dtype=np.float64)
    t = total_samples
    p = n_features
    df = float(p + (p * (p + 1)) // 2)

    s_tot = total_sum
    sxx_tot = total_sum_outer

    sigma_tot = (sxx_tot - np.outer(s_tot, s_tot) / t) / t
    sign_tot, logdet_tot = np.linalg.slogdet(sigma_tot)
    if sign_tot <= 0:
        return out

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1

        if n1 <= 2 * p or n2 <= 2 * p:
            out[i] = 0.0
            continue

        s1 = before_sums[i]
        sxx1 = before_sum_outers[i]
        s2 = s_tot - s1
        sxx2 = sxx_tot - sxx1

        sigma_1 = (sxx1 - np.outer(s1, s1) / n1) / n1
        sigma_2 = (sxx2 - np.outer(s2, s2) / n2) / n2

        sign1, logdet1 = np.linalg.slogdet(sigma_1)
        sign2, logdet2 = np.linalg.slogdet(sigma_2)
        if sign1 <= 0 or sign2 <= 0:
            out[i] = 0.0
            continue

        out[i] = t * logdet_tot - n1 * logdet1 - n2 * logdet2 - df

    return out


@nb.njit(cache=True)
def reference_multivariate_mean_or_covariance_penalty(
    n_samples: int, n_features: int
) -> float:
    """Legacy mean-or-covariance penalty ``sqrt(df log t) + log t``."""
    p = n_features
    df = float(p + (p * (p + 1)) // 2)
    return np.sqrt(df * np.log(n_samples)) + np.log(n_samples)


@nb.njit(cache=True)
def _reference_scalar_max_ll(k_i: float, n_i: int) -> float:
    if k_i <= 0.0 or k_i >= n_i:
        return 0.0
    p_hat = k_i / n_i
    return k_i * np.log(p_hat) + (n_i - k_i) * np.log1p(-p_hat)


@nb.njit(cache=True)
def _reference_bernoulli_max_ll(k: np.ndarray, n: int) -> np.ndarray:
    out = np.empty(k.shape, dtype=np.float64)
    for i in range(k.shape[0]):
        out[i] = _reference_scalar_max_ll(k[i], n)
    return out


@nb.njit(cache=True)
def reference_npfocus_score(
    total_sum: np.ndarray,
    before_sums: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> np.ndarray:
    """Legacy NPFOCuS score: ``(G, 2)`` channel-max of [grid-sum, grid-max]."""
    n_candidates = before_sums.shape[0]
    scores = np.zeros((n_candidates, 2), dtype=np.float64)
    if total_samples <= 0:
        return scores

    n_features = total_sum.shape[0]

    for j in range(n_candidates):
        n_pre = before_samples[j]
        n_post = total_samples - n_pre
        if n_pre <= 0 or n_post <= 0:
            continue

        best_sum_score = 0.0
        best_max_score = 0.0
        for feature_idx in range(n_features):
            total_sum_feature = total_sum[feature_idx]
            sum_pre_feature = before_sums[j, feature_idx]
            sum_post_feature = total_sum_feature - sum_pre_feature

            ll_tot = _reference_bernoulli_max_ll(total_sum_feature, total_samples)
            ll_pre = _reference_bernoulli_max_ll(sum_pre_feature, n_pre)
            ll_post = _reference_bernoulli_max_ll(sum_post_feature, n_post)
            lr_scores = 2.0 * (ll_post + ll_pre - ll_tot)

            sum_score = np.sum(lr_scores)
            max_score = np.max(lr_scores)
            if feature_idx == 0 or sum_score > best_sum_score:
                best_sum_score = sum_score
            if feature_idx == 0 or max_score > best_max_score:
                best_max_score = max_score

        scores[j, 0] = best_sum_score
        scores[j, 1] = best_max_score

    return scores
