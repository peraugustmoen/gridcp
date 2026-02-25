import numpy as np
import numba as nb
import math
from .utils import fastlog


#### Univariate Mean Change with known variance ####
@nb.njit
def h_univariate_mean_known_var_LR(y):
    return y


@nb.njit
def f_univariate_mean_known_var_LR(sum_pre_j, sum_post_j, g, t):
    res = math.sqrt(1.0 * g / (t * (t - g))) * sum_pre_j[0]
    res = res - math.sqrt(1.0 * (t - g) / t / g) * sum_post_j[0]
    return res * res - 1


@nb.njit
def penalty_univariate_mean_known_var_LR(g, t, p):
    logg = fastlog(t)
    logg = logg + math.sqrt(logg)
    return logg


#### Univariate Mean Change with unknown variance ####
@nb.njit
def h_univariate_mean_unknown_var_LR(y):
    ret = np.empty(2, dtype=np.float64)
    ret[0] = y[0]
    ret[1] = y[0] * y[0]
    return ret


@nb.njit
def f_univariate_mean_unknown_var_LR(sum_pre_j, sum_post_j, g, t):
    n1 = t - g
    n2 = g

    if n1 <= 2 or n2 <= 2:
        return 0.0
    totalsum = sum_pre_j + sum_post_j
    sum_pre_j_id = sum_pre_j[0]
    sum_post_j_id = sum_post_j[0]
    totalsum_id = totalsum[0]

    sum_pre_j_var = sum_pre_j[1]
    sum_post_j_var = sum_post_j[1]
    totalsum_var = totalsum[1]

    Sigma_null = (totalsum_var - totalsum_id * totalsum_id / t) / t
    Sigma_alt = (
        (sum_pre_j_var - sum_pre_j_id * sum_pre_j_id / n1)
        + (sum_post_j_var - sum_post_j_id * sum_post_j_id / n2)
    ) / t

    LR = t * math.log(Sigma_null) - t * math.log(Sigma_alt)
    df = 1  # Number of parameters in mean+covariance

    return LR - df


@nb.njit
def penalty_univariate_mean_unknown_var_LR(g, t, p):
    logg = fastlog(2 * t)
    logg = logg + math.sqrt(logg)
    return logg


#### Univariate Variance Change ####
@nb.njit
def h_univariate_variance_LR(y):
    return y * y


@nb.njit
def f_univariate_variance_LR(sum_pre_j, sum_post_j, g, t):
    n1 = t - g
    n2 = g

    totalsum = sum_pre_j[0] + sum_post_j[0]

    Sigma_tot = (totalsum) / t
    Sigma_pre_j = sum_pre_j[0] / n1
    Sigma_post_j = sum_post_j[0] / n2

    LR = (
        t * math.log(Sigma_tot)
        - n1 * math.log(Sigma_pre_j)
        - n2 * math.log(Sigma_post_j)
    )
    df = 1  # Number of parameters in mean+covariance

    return LR - df


@nb.njit
def penalty_univariate_variance_LR(g, t, p):
    logg = fastlog(t)
    logg = logg + math.sqrt(logg)
    return logg


#### Univariate Variance Change unknown mean ####
## ... this does not have a closed-form solution...


#### Univariate Mean or Variance Change ####
@nb.njit
def h_univariate_mean_or_variance_LR(y):
    ret = np.empty(2, dtype=np.float64)
    ret[0] = y[0]
    ret[1] = y[0] * y[0]
    return ret


@nb.njit
def f_univariate_mean_or_variance_LR(sum_pre_j, sum_post_j, g, t):
    n1 = t - g
    n2 = g
    if n1 <= 2 or n2 <= 2:
        return 0.0

    totalsum = sum_pre_j + sum_post_j
    sum_pre_j_id = sum_pre_j[0]
    sum_post_j_id = sum_post_j[0]
    totalsum_id = totalsum[0]

    sum_pre_j_var = sum_pre_j[1]
    sum_post_j_var = sum_post_j[1]
    totalsum_var = totalsum[1]

    Sigma_tot = (totalsum_var - totalsum_id * totalsum_id / t) / t
    Sigma_pre_j = (sum_pre_j_var - sum_pre_j_id * sum_pre_j_id / n1) / n1
    Sigma_post_j = (sum_post_j_var - sum_post_j_id * sum_post_j_id / n2) / n2

    LR = (
        t * math.log(Sigma_tot)
        - n1 * math.log(Sigma_pre_j)
        - n2 * math.log(Sigma_post_j)
    )
    df = 2  # Number of parameters in mean+covariance

    return LR - df


@nb.njit
def penalty_univariate_mean_or_variance_LR(g, t, p):
    logg = fastlog(2 * t)
    logg = logg + math.sqrt(logg)
    return logg


## Multivariate Change in mean, identity covariance:
@nb.njit
def h_multivariate_mean_id_cov_LR(y):
    return y


@nb.njit
def f_multivariate_mean_id_cov_LR(sum_pre_j, sum_post_j, g, t):
    n1 = t - g
    n2 = g
    # segment means:
    mean1 = sum_pre_j / n1
    mean2 = sum_post_j / n2
    diff = mean1 - mean2
    # LR statistic ~ chi^2_p:
    T = g * (t - g) / (1.0 * t) * np.dot(diff, diff)
    df = diff.shape[0]  # p
    return T - df


@nb.njit
def penalty_multivariate_mean_id_cov_LR(g, t, p):
    logg = fastlog(t)
    logg = logg + math.sqrt(p * logg)
    return logg


## Multivariate Change in mean, unkonwn covariance:
@nb.njit
def h_multivariate_mean_unknown_cov_LR(y):
    """
    Sufficient statistic for Gaussian mean+covariance:
    Return concatenation of y and vec(y y^T).

    If y has shape (p,), this returns an array of shape (p + p*p,)
    where the last p*p entries are column-major flattening of y y^T.
    """
    p = y.shape[0]
    yy = np.outer(y, y)
    out = np.empty((p + 1, p), dtype=y.dtype)
    out[0] = y
    out[1:] = yy

    return out


@nb.njit
def f_multivariate_mean_unknown_cov_LR(sum_pre_j, sum_post_j, g, t):
    """
    GLR-type statistic for a change in both mean and covariance
    in multivariate Gaussian data.

    sum_pre_j: sum of h(y) over segment 1  (shape (p+1, p))
    sum_post_j: sum of h(y) over segment 2 (shape (p+1, p))
    g: candidate change-point (segment 1 length)
    t: total sample size
    """
    n1 = t - g
    n2 = g

    p = sum_pre_j.shape[0] - 1  # dimension of data

    totalsum = sum_pre_j + sum_post_j
    sum_pre_j_id = sum_pre_j[0]
    sum_post_j_id = sum_post_j[0]
    totalsum_id = totalsum[0]

    sum_pre_j_cov = sum_pre_j[1:]
    sum_post_j_cov = sum_post_j[1:]
    totalsum_cov = totalsum[1:]

    Sigma_null = (totalsum_cov - np.outer(totalsum_id, totalsum_id) / t) / t
    Sigma_alt = (
        (sum_pre_j_cov - np.outer(sum_pre_j_id, sum_pre_j_id) / n1)
        + (sum_post_j_cov - np.outer(sum_post_j_id, sum_post_j_id) / n2)
    ) / t

    sign0, logdet0 = np.linalg.slogdet(Sigma_null)
    sign1, logdet1 = np.linalg.slogdet(Sigma_alt)
    if sign0 <= 0 or sign1 <= 0:
        return 0.0

    LR = t * (logdet0 - logdet1)
    df = p  # Number of df

    return LR - df


@nb.njit
def penalty_multivariate_mean_unknown_cov_LR(g, t, p):
    df = p
    logg = fastlog(t / 0.05)
    rr = math.sqrt(df * logg) + logg

    return rr


#### Multivariate Change in Mean or Covariance ####
@nb.njit
def h_multivariate_mean_and_covariance_LR(y):
    """
    Sufficient statistic for Gaussian mean+covariance:
    Return concatenation of y and vec(y y^T).

    If y has shape (p,), this returns an array of shape (p + p*p,)
    where the last p*p entries are column-major flattening of y y^T.
    """
    p = y.shape[0]
    yy = np.outer(y, y)
    out = np.empty((p + 1, p), dtype=y.dtype)
    out[0] = y
    out[1:] = yy

    return out


@nb.njit
def f_multivariate_mean_and_covariance_LR(sum_pre_j, sum_post_j, g, t):
    """
    GLR-type statistic for a change in both mean and covariance
    in multivariate Gaussian data.

    sum_pre_j: sum of h(y) over segment 1  (shape (p+1, p))
    sum_post_j: sum of h(y) over segment 2 (shape (p+1, p))
    g: candidate change-point (segment 1 length)
    t: total sample size
    """
    n1 = t - g
    n2 = g

    p = sum_pre_j.shape[0] - 1  # dimension of data
    if n1 <= (p + 1) or n2 <= (p + 1):
        return 0.0

    totalsum = sum_pre_j + sum_post_j
    sum_pre_j_id = sum_pre_j[0]
    sum_post_j_id = sum_post_j[0]
    totalsum_id = totalsum[0]

    sum_pre_j_cov = sum_pre_j[1:]
    sum_post_j_cov = sum_post_j[1:]
    totalsum_cov = totalsum[1:]

    Sigma_tot = (totalsum_cov - np.outer(totalsum_id, totalsum_id) / t) / t
    Sigma_pre_j = (sum_pre_j_cov - np.outer(sum_pre_j_id, sum_pre_j_id) / n1) / n1
    Sigma_post_j = (sum_post_j_cov - np.outer(sum_post_j_id, sum_post_j_id) / n2) / n2

    # GLR statistic: t * log|Sigma| - g* log|Sigma1| - (t-g) * log|Sigma2|
    # Use slogdet for numerical stability
    sign0, logdet0 = np.linalg.slogdet(Sigma_tot)
    sign1, logdet1 = np.linalg.slogdet(Sigma_pre_j)
    sign2, logdet2 = np.linalg.slogdet(Sigma_post_j)

    LR = t * logdet0 - n1 * logdet1 - n2 * logdet2
    df = p + (p * (p + 1)) // 2  # Number of parameters in mean+covariance

    return LR - df


@nb.njit
def penalty_multivariate_mean_and_covariance_LR(g, t, p):
    df = (p * (p + 1)) // 2 + p
    logg = fastlog(t / 0.05)
    rr = math.sqrt(df * logg) + logg

    return rr
