import numpy as np
from scipy.stats import nbinom, poisson, norm
from joblib import Parallel, delayed
from gglasso.problem import glasso_problem


def re_parameterize_nb(mu, theta):
    """
    Re-parameterize the parameters of a negative binomial distribution.

    This function converts the mean and dispersion parameters of a negative binomial distribution 
    to the parameters used by numpy's nbinom function.

    Parameters:
    - mu (float or array-like): The mean parameter(s) of the negative binomial distribution.
    - theta (float or array-like): The dispersion parameter(s) of the negative binomial distribution.

    Returns:
    - tuple: A tuple containing two elements:
        - n (float or array-like): The number of successes until the experiment is stopped.
        - p (float or array-like): The probability of success in each experiment.
    """
    var = mu + (1 / theta) * mu ** 2
    p = mu / var
    n = mu ** 2 / (var - mu)

    return n, p


def get_distribution(mu, theta):
    """
    Get the distribution based on the provided mean and dispersion parameters.

    This function returns a Poisson distribution if the dispersion parameter `theta` is infinity.
    Otherwise, it returns a negative binomial distribution.

    Parameters:
    - mu (float): The mean parameter of the distribution.
    - theta (float): The dispersion parameter of the distribution. If theta is infinity, a Poisson distribution is used. Otherwise, a negative binomial distribution is used.

    Returns:
    - dist (scipy.stats._distn_infrastructure.rv_frozen): The selected distribution. This is an instance of a frozen distribution object from scipy.stats, which can be used to calculate the probability density function (PDF), cumulative distribution function (CDF), and other properties of the distribution.
    """
    if theta == np.inf:
        dist = poisson(mu=mu)
    else:
        n, p = re_parameterize_nb(mu, theta)
        dist = nbinom(n, p)

    return dist


def scale_data(X, zero_prob):
    """
    Scale the data based on the zero inflation probability.

    This function scales the data `X` to the range `[0, 1]` by subtracting `zero_prob` 
    and dividing by `1 - zero_prob`. If `zero_prob` is 0, the data is not scaled.

    Parameters:
    - X (ndarray): A 1D or 2D numpy array representing the data to be scaled.
    - zero_prob (float): The zero inflation probability. Must be in the range `[0, 1]`.

    Returns:
    - ndarray: The scaled data. Has the same shape as `X`.
    """
    if zero_prob != 0:
        one_minus_zero_prob = 1 - zero_prob
        X = (X - zero_prob) / one_minus_zero_prob
    return X


def select_cdf(X, mu, theta, zero_prob):
    """
    Select the cumulative distribution function (CDF) for a given distribution.
    If zero_prob is not zero, the CDF is adjusted to account for zero inflation. 
    The CDF is scaled down by a factor of (1 - zero_prob) because the non-zero part of the distribution now only accounts for (1 - zero_prob) proportion of the data. 
    Then, zero_prob is added to the scaled CDF to account for the additional probability of zeros.

    Parameters:
    - X (array-like): The values at which to calculate the CDF.
    - mu (float): The mean parameter of the distribution.
    - theta (float): The dispersion parameter of the distribution. If theta is infinity, a Poisson distribution is used. Otherwise, a negative binomial distribution is used.
    - zero_prob (float): The probability of zero. If zero_prob is not zero, it's added to the CDF.

    Returns:
    - array-like: The selected CDF.
    """
    dist = get_distribution(mu, theta)
    cdf = dist.cdf(X)

    if zero_prob != 0:
        cdf = zero_prob + (1 - zero_prob) * cdf

    return cdf
        

def select_inv_cdf(X, mu, theta, zero_prob):
    """
    Select the inverse cumulative distribution function (PPF) for a given distribution.
    If zero_prob is not zero, the data `X` is scaled to the range `[0, 1]` by subtracting `zero_prob` and dividing by `1 - zero_prob`. 
    This is done because the non-zero part of the distribution now only accounts for `(1 - zero_prob)` proportion of the data. 
    Then, zero_prob is added to the PPF to account for the additional probability of zeros.

    Parameters:
    - X (array-like): The values at which to calculate the PPF.
    - mu (float): The mean parameter of the distribution.
    - theta (float): The dispersion parameter of the distribution. If theta is infinity, a Poisson distribution is used. Otherwise, a negative binomial distribution is used.
    - zero_prob (float): The probability of zero. If zero_prob is not zero, it's used to scale the data and adjust the PPF.

    Returns:
    - array-like: The selected PPF.
    """

    if zero_prob != 0:
        X = scale_data(X, zero_prob)

    dist = get_distribution(mu, theta)
    inv_cdf = dist.ppf(X)

    if zero_prob != 0:
        inv_cdf[X < zero_prob] = 0

    return inv_cdf


def calculate_cdf(X, marginal, i):
    """
    Calculate the cumulative distribution function (CDF) for a given species.

    This function calculates the CDF for the counts of a given species based on the parameters of the marginal distribution for that species.

    Parameters:
    - X (DataFrame): A DataFrame where each row represents a species and each column represents a sample.
    - marginal (dict): A dictionary containing the parameters of the marginal distribution for each species.
    - i (int): The index of the species for which to calculate the CDF.

    Returns:
    - array-like: The calculated CDF values for the given species.
    """
    zero_prob = marginal["params"][i][0]
    theta = marginal["params"][i][1]
    mu = marginal["params"][i][2]

    counts = X.values[i, :]
    cdf = select_cdf(counts, mu, theta, zero_prob)

    return cdf


def calculate_inv_cdf(X, marginal, i):
    """
    Calculate the inverse cumulative distribution function (CDF) for a given distribution and data.

    Parameters:
    - X (np.array): The data for which to calculate the inverse CDF. Each row corresponds to a different distribution.
    - marginal (dict): A dictionary containing the parameters for each distribution. The "params" key should map to a list of tuples, where each tuple contains the parameters for a distribution in the order (zero_prob, theta, mu).
    - i (int): The index of the distribution to use.

    Returns:
    - array-like: The inverse CDF for the i-th distribution and the i-th row of `X`.

    The function extracts the parameters for the i-th distribution from `marginal`, then calculates the inverse CDF for the i-th row of `X` using these parameters.
    """

    zero_prob = marginal["params"][i][0]
    theta = marginal["params"][i][1]
    mu = marginal["params"][i][2]

    counts = X[i, :]
    inv_cdf = select_inv_cdf(counts, mu, theta, zero_prob)

    return inv_cdf


def marginal_to_unifrom(X, marginal):
    """
    Calculates the cumulative distribution function (CDF) for each species in the given data in parallel.

    Parameters:
    - X (DataFrame): A DataFrame where each row represents a species and each column represents a sample.
    - marginal (dict): A dictionary containing the parameters of the marginal distribution for each species.

    Returns:
    - ndarray: A 2D numpy array where each row represents a species and each column represents a sample. The values correspond to CDF values.
    """
    p, _ = X.shape

    uniform_counts = Parallel(n_jobs=-1)(delayed(calculate_cdf)(X, marginal, i) for i in range(p))

    return np.array(uniform_counts)


def uniform_to_marginal(Z, marginal):
    """
    Calculate the inverse cumulative distribution function (CDF) for each species in the given data in parallel.

    Parameters:
    - Z (ndarray): A 2D numpy array where each row represents a species and each column represents a sample. The values are the results of the distribution transformation.
    - marginal (dict): A dictionary containing the parameters of the marginal distribution for each species.

    Returns:
    - ndarray: A 2D numpy array where each row represents a species and each column represents a sample. The values are the results of the inverse distribution transformation.
    """
    p, _ = Z.shape

    counts = Parallel(n_jobs=-1)(delayed(calculate_inv_cdf)(Z, marginal, i) for i in range(p))

    return np.array(counts)


def distribution_transform(X, marginal, rng):
    """
    Perform a distribution transformation on the given data.

    This function calculates the cumulative distribution function (CDF) of the data `X` under the given marginal distribution (F), 
    and the CDF of `X+1` (F1). It then generates a 2D array of random numbers uniformly distributed between 0 and 1 (V), 
    and calculates a weighted mixture of `F` and `F1` (U). The weights are the random numbers. This is the actual distribution transformation.

    Parameters:
    - X (DataFrame): A DataFrame where each row represents a species and each column represents a sample.
    - marginal (dict): A dictionary containing the parameters of the marginal distribution for each species.
    - rng (Generator): A numpy random number generator.

    Returns:
    - U (ndarray): A 2D numpy array where each row represents a species and each column represents a sample. The values are the results of the distribution transformation.

    Intermediate Variables:
    - F (ndarray): The CDF of the data `X` under the given marginal distribution.
    - F1 (ndarray): The CDF of `X+1` under the given marginal distribution.
    - V (ndarray): A 2D array of random numbers uniformly distributed between 0 and 1. The shape of the array is the same as `F`.
    """
    F = marginal_to_unifrom(X, marginal)
    F1 = marginal_to_unifrom(X+1, marginal)

    V = rng.uniform(0, 1, F.shape)
    U = V * F + (1 - V) * F1

    return U

def check_and_replace(U, eps=1e-10):
    """
    Check if the array `U` contains extreme values of 0 or 1 and replace them.

    This function checks if the array `U` contains any values of 0 or 1 and replaces them with `eps` and `1 - eps` respectively.

    Parameters:
    - U (ndarray): A 2D numpy array where each row represents a species and each column represents a sample.
    - eps (float): A small value that 0 will be replaced with and 1 will be replaced with `1 - eps`.

    Returns:
    - U (ndarray): The corrected array where values of 0 and 1 have been replaced.
    """
    U = np.where(U == 0, eps, U)
    U = np.where(U == 1, 1 - eps, U)

    return U


def estimate_covariance(X, method="scaled"):
    """
    Estimate the covariance matrix of the given data using the specified method.

    This function supports three methods for estimating the covariance matrix:
    - "scaled": Scales the data and then calculates the correlation coefficient matrix.
    - "unscaled": Calculates the covariance matrix without scaling the data.
    - "inverse": Estimates the inverse covariance matrix using the graphical lasso algorithm.

    Parameters:
    - X (ndarray): A 2D numpy array where each row represents a species and each column represents a sample.
    - method (str): The method to use for estimating the covariance matrix. Must be one of "scaled", "unscaled", or "inverse". Default is "scaled".

    Returns:
    - R_est (ndarray): The estimated covariance matrix.
    """
    if method == "scaled":
        R_est = np.corrcoef(X)
    elif method == "unscaled":
        R_est = np.cov(X, bias=True)
    elif method == "inverse":
        S = np.cov(X)
        N = X.shape[1]
        # TO DO: later use function from q2-gglasso for etimating the precision matrix
        P = glasso_problem(S, N, latent = False, do_scaling = False)
        lambda1_range = np.logspace(0, -3, 30)
        modelselect_params = {'lambda1_range': lambda1_range}
        P.model_selection(modelselect_params = modelselect_params, method = 'eBIC', gamma = 0.1)
        R_est = P.solution.precision_

    return R_est


def estimate_null_parameters(X, marginal, rng, cov_method="scaled"):
    """
    Fit a Gaussian copula to the given data.

    Parameters:
    - X (DataFrame): A DataFrame where each row represents a species and each column represents a sample.
    - marginal (dict): A dictionary containing the parameters of the marginal distribution for each species.

    Returns:
    - ndarray: A 2D numpy array where each row represents a species and each column represents a sample. The values are the results of the distribution transformation.

    This function fits a Gaussian copula to the given data based on the parameters of the marginal distribution for each species. It then performs a distribution transformation on the data and returns the results.
    """
    
    U = distribution_transform(X, marginal, rng)

    U = check_and_replace(U)

    U_inv = norm.ppf(U, 0, 1)

    R_est = estimate_covariance(U_inv, cov_method)

    param_dict = {"U": U, "U_inv": U_inv, "R_est": R_est}

    return param_dict