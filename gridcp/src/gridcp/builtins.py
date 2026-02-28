import numpy as np
import numba as nb
import math
from .utils import fastlog, logdet_spd, inv_sqrtm_pd


#### Univariate Mean Change with known variance ####
@nb.njit(cache=True)
def h_univariate_mean_known_var_LR(y):
    return y


@nb.njit(cache=True)
def f_univariate_mean_known_var_LR(sum_pre_j, sum_post_j, pos, t):
    # pos = t-g
    g = t - pos
    res = math.sqrt(1.0 * g / (t * (t - g))) * sum_pre_j[0]
    res = res - math.sqrt(1.0 * (t - g) / t / g) * sum_post_j[0]
    return res * res - 1


@nb.njit(cache=True)
def penalty_univariate_mean_known_var_LR(pos, t, p):
    logg = fastlog(t)
    logg = logg + math.sqrt(logg)
    return logg


#### Univariate Mean Change with unknown variance ####
@nb.njit(cache=True)
def h_univariate_mean_unknown_var_LR(y):
    ret = np.empty(2, dtype=np.float64)
    ret[0] = y[0]
    ret[1] = y[0] * y[0]
    return ret


@nb.njit(cache=True)
def f_univariate_mean_unknown_var_LR(sum_pre_j, sum_post_j, pos, t):
    g = t - pos
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


@nb.njit(cache=True)
def penalty_univariate_mean_unknown_var_LR(pos, t, p):
    logg = fastlog(2 * t)
    logg = logg + math.sqrt(logg)
    return logg


#### Univariate Variance Change ####
@nb.njit(cache=True)
def h_univariate_variance_LR(y):
    return y * y


@nb.njit(cache=True)
def f_univariate_variance_LR(sum_pre_j, sum_post_j, pos, t):
    g = t - pos
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


@nb.njit(cache=True)
def penalty_univariate_variance_LR(pos, t, p):
    logg = fastlog(t)
    logg = logg + math.sqrt(logg)
    return logg


#### Univariate Variance Change unknown mean ####
## ... this does not have a closed-form solution...


#### Univariate Mean or Variance Change ####
@nb.njit(cache=True)
def h_univariate_mean_or_variance_LR(y):
    ret = np.empty(2, dtype=np.float64)
    ret[0] = y[0]
    ret[1] = y[0] * y[0]
    return ret


@nb.njit(cache=True)
def f_univariate_mean_or_variance_LR(sum_pre_j, sum_post_j, pos, t):
    g = t - pos
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


@nb.njit(cache=True)
def penalty_univariate_mean_or_variance_LR(pos, t, p):
    logg = fastlog(2 * t)
    logg = logg + math.sqrt(logg)
    return logg


## Multivariate Change in mean, identity covariance:
@nb.njit(cache=True)
def h_multivariate_mean_id_cov_LR(y):
    return y


@nb.njit(cache=True)
def f_multivariate_mean_id_cov_LR(sum_pre_j, sum_post_j, pos, t):
    g = t - pos
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
    # res = math.sqrt(1.0 * (t - pos) / (t * pos)) * sum_pre_j
    # res = res - math.sqrt(1.0 * pos / t / (t - pos)) * sum_post_j
    # df = res.shape[0]
    # return np.sum(res * res) - df


@nb.njit(cache=True)
def penalty_multivariate_mean_id_cov_LR(pos, t, p):
    logg = fastlog(t)
    logg = logg + math.sqrt(p * logg)
    return logg


## Multivariate Change in mean, unkonwn covariance:
@nb.njit(cache=True)
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


@nb.njit(cache=True)
def f_multivariate_mean_unknown_cov_LR(sum_pre_j, sum_post_j, pos, t):
    """
    GLR-type statistic for a change in both mean and covariance
    in multivariate Gaussian data.

    sum_pre_j: sum of h(y) over segment 1  (shape (p+1, p))
    sum_post_j: sum of h(y) over segment 2 (shape (p+1, p))
    g: candidate change-point (segment 1 length)
    t: total sample size
    """
    g = t - pos
    n1 = t - g
    n2 = g

    p = sum_pre_j.shape[0] - 1  # dimension of data

    if t < 2 * p:
        return 0.0
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


@nb.njit(cache=True)
def penalty_multivariate_mean_unknown_cov_LR(pos, t, p):
    df = p
    logg = fastlog(t / 0.05)
    rr = math.sqrt(df * logg) + logg

    return rr


#### Multivariate Change in Mean or Covariance ####
@nb.njit(cache=True)
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


@nb.njit(cache=True)
def f_multivariate_mean_and_covariance_LR(sum_pre_j, sum_post_j, pos, t):
    """
    GLR-type statistic for a change in both mean and covariance
    in multivariate Gaussian data.

    sum_pre_j: sum of h(y) over segment 1  (shape (p+1, p))
    sum_post_j: sum of h(y) over segment 2 (shape (p+1, p))
    pos: candidate change-point (segment 1 length)
    t: total sample size
    """
    g = t - pos
    n1 = t - g
    n2 = g

    p = sum_pre_j.shape[0] - 1  # dimension of data
    if n1 <= 2 * p or n2 <= 2 * p:
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


@nb.njit(cache=True)
def penalty_multivariate_mean_and_covariance_LR(pos, t, p):
    df = (p * (p + 1)) // 2 + p
    logg = fastlog(t / 0.05)
    rr = math.sqrt(df * logg) + logg

    return rr


### Regression
@nb.njit(cache=True)
def h_regression_mcscan(y):
    """
    First entry of Y is response, next entries are covariates
    """
    return y[0] * y[1:]


@nb.njit(cache=True)
def f_regression_mcscan(sum_pre_j, sum_post_j, pos, t):
    """
    GLR-type statistic for a change in both mean and covariance
    in multivariate Gaussian data.

    sum_pre_j: sum of h(y) over segment 1  (shape (p+1, p))
    sum_post_j: sum of h(y) over segment 2 (shape (p+1, p))
    pos: candidate change-point (segment 1 length)
    t: total sample size
    """

    g = t - pos
    cov1 = sum_pre_j / (t - g)
    cov2 = sum_post_j / g
    dist = np.max(np.abs(cov1 - cov2))
    ret = math.sqrt(g * (t - g) / (1.0 * t)) * dist

    return ret


@nb.njit(cache=True)
def penalty_regression_mcscan(pos, t, p):
    # this is the penalty under Assumption 2(ii) in McScan paper Cho, Kley, Li (2025, JRSS-B)
    rr = math.sqrt(math.log(p * t))

    return rr


@nb.njit(cache=True)
def gen_gaussian_regression_obs(beta0, add_perturbation=False, rho=0, s=1):
    ## This func generates instances from the
    ## simulation setup in M1 in McScan paper Cho, Kley, Li (2025, JRSS-B).
    ## We generate data \eps_i \sim N(0,1),
    ## X_i \sim N(0, I), Y_i = X_i^T beta0 + \eps_i.
    ## If rho>0, then an s-sparse perturbation is added to
    ## beta0, sampled from the sphere.
    ## The output is a vector of length p+1, where the first entry is Y_i and the next p entries are X_i.

    q = len(beta0)
    out = np.zeros(q + 1)
    out[1:] = np.random.normal(loc=0, scale=1, size=q)
    beta = np.copy(beta0)

    if add_perturbation and rho > 0:
        beta_pert = np.random.normal(loc=0, scale=1, size=s)
        # Normalize to have magnitude rho
        beta_pert = beta_pert * rho / math.sqrt(np.sum(beta_pert * beta_pert))
        beta[:s] = beta[:s] + beta_pert

    out[0] = (np.dot(out[1:], beta) + np.random.normal(loc=0, scale=s, size=1))[0]
    return out


### direct method
@nb.njit(cache=True)
def h_regression_direct(y):
    """
    First entry of Y is response, next entries are covariates
    """
    q = y.shape[0] - 1
    out = np.zeros(q + q * q, dtype=np.float64)
    out[0:q] = y[0] * y[1:]
    out[q:] = np.outer(y[1:], y[1:]).flatten()
    return out


@nb.njit(cache=True)
def f_regression_direct(sum_pre_j, sum_post_j, pos, t):
    """
    GLR-type statistic for a change in both mean and covariance
    in multivariate Gaussian data.

    sum_pre_j: sum of h(y) over segment 1  (shape (p+1, p))
    sum_post_j: sum of h(y) over segment 2 (shape (p+1, p))
    pos: candidate change-point (segment 1 length)
    t: total sample size
    """
    v = sum_pre_j.shape[0]
    q = int(math.sqrt(1 + 4 * v) - 1) // 2  # number of covariates
    g = t - pos
    n1 = t - g
    n2 = g

    if n1 <= q or n2 <= q:
        return 0.0
    yxt_pre = sum_pre_j[:q]
    yxt_post = sum_post_j[:q]
    xx_pre = sum_pre_j[q:].reshape((q, q))
    xx_post = sum_post_j[q:].reshape((q, q))

    M1_inv = inv_sqrtm_pd(xx_pre)
    M2_inv = inv_sqrtm_pd(xx_post)
    diff = M1_inv @ yxt_pre - M2_inv @ yxt_post
    normdiff_sq = np.sum(diff * diff)

    return normdiff_sq / 2 - q


@nb.njit(cache=True)
def penalty_regression_direct(pos, t, p):
    # this is the penalty under Assumption 2(ii) in McScan paper Cho, Kley, Li (2025, JRSS-B)
    rr = math.sqrt((p - 1) * math.log(t)) + math.log(t)

    return rr
