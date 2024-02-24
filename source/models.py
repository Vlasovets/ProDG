import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, logistic, nbinom


def fit_poisson(x, intercept, pval_cutoff):
    """
    Fit a Poisson distribution or a zero-inflated Poisson model to a given array.

    Parameters:
    - x (array-like): An array of count data.
    - intercept (array-like): An array of intercept values for the model.
    - pval_cutoff (float): The p-value cutoff for the likelihood ratio test.

    Returns:
    - list: A list containing the parameters [zero_prob, theta, lambda] of the fitted model.
    - zero_prob (float): Probability of observing zero counts.
    - theta (float): Dispersion parameter (inverse of alpha) for the Poisson or zero-inflated Poisson model.
    - lambda (float): Mean parameter for the Poisson model.

    This function fits either a Poisson distribution or a zero-inflated Poisson model to the count data `x` based on the likelihood ratio test. If the zero-inflated model is preferred, it returns the parameters for that model. Otherwise, it returns the parameters for a Poisson distribution.

    Reference: https://www.statsmodels.org/stable/discretemod.html
    """

    mle_poisson = sm.GLM(x, intercept, family=sm.families.Poisson()).fit()
    
    try:
        mle_zip = sm.ZeroInflatedPoisson(x, intercept).fit()
        chisq_val = 2 * (mle_zip.llf - mle_poisson.llf)
        pvalue = 1 - chi2.cdf(chisq_val, 1)

        if pvalue < pval_cutoff:
            return [logistic.cdf(mle_zip.params['inflate_const']), np.inf, np.exp(mle_zip.params['const'])]
        else:
            return [0.0, np.inf, np.mean(x)]
        
    except Exception:
        return [0.0, np.inf, np.mean(x)]


def fit_nb(x):
    """
    Fit a negative binomial distribution to a given array.

    Parameters:
    - x (array-like): An array of count data.

    Returns:
    - list: A list containing the parameters [zero_prob, theta, lambda] of the fitted negative binomial model.
    - zero_prob (float): Probability of observing zero counts.
    - theta (float): Dispersion parameter (inverse of alpha) for the negative binomial distribution.
    - lambda (float): Mean parameter for the negative binomial distribution.

    If the data suggests a Poisson distribution is a better fit, it returns the parameters for the Poisson distribution. 
    Otherwise, it returns the parameters for the negative binomial distribution.

    Reference: https://www.statsmodels.org/stable/discretemod.html
    """
    mu, var = np.mean(x), np.var(x)
    intercept = sm.add_constant(np.ones(len(x)))
    
    if mu >= var: 
        # Poisson
        return [0.0, np.inf, mu]
    
    else:
        # Negative binomial
        mle_nb = sm.NegativeBinomial(x, exog=intercept).fit()
        theta_nb = 1 / mle_nb.params['alpha']
                    
        return [0.0, theta_nb, np.exp(mle_nb.params['const'])]


def fit_zinb(x, method='nm', maxiter=5000, gtol=1e-12, pval_cutoff=0.05):
    """
    Fit a zero-inflated negative binomial (ZINB) model or related models to a given count data array.

    Parameters:
    - x (array-like): An array of count data.
    - method (str): The method to use for fitting the model. Default is 'nm'.
    - maxiter (int): The maximum number of iterations. Default is 5000.
    - gtol (float): The gradient tolerance. Default is 1e-12.
    - pval_cutoff (float): The p-value cutoff for the likelihood ratio test. Default is 0.05.

    Returns:
    - list: A list containing the parameters [zero_prob, theta, lambda] of the fitted model, or the result of fit_poisson or fit_nb function.
        - zero_prob (float): Probability of observing zero counts.
        - theta (float): Dispersion parameter (inverse of alpha) for the negative binomial component.
        - lambda (float): Mean parameter for the ZINB model.

    This function fits a ZINB model to the count data `x` if the data suggests it is a better fit. It uses a likelihood ratio test to determine if the zero-inflated model is needed. If a ZINB model is not needed, it falls back to a Poisson or negative binomial model based on the data characteristics.

    Reference: https://www.statsmodels.org/stable/discretemod.html
    """
    
    mu, var = np.mean(x), np.var(x)
    intercept = sm.add_constant(np.ones(len(x)))
    
    if mu >= var:
        return fit_poisson(x, intercept, pval_cutoff)
    else:
        if np.min(x) > 0:
            return fit_nb(x)
        else:
            try:

                mle_zinb = sm.ZeroInflatedNegativeBinomialP(x, intercept).fit(method=method, maxiter=maxiter, gtol=gtol)
                theta_zinb = 1 / mle_zinb.params['alpha']
                    
                return [logistic.cdf(mle_zinb.params['inflate_const']), theta_zinb, np.exp(mle_zinb.params['const'])]
            
            except Exception:
                return fit_nb(x)
            

def fit_auto(x, pval_cutoff, method_zinb='nm', maxiter_zinb=5000, gtol_zinb=1e-12):
    """
    Fit parameters for a probability distribution to a given array of count data.

    Parameters:
    - x (array-like): An array of count data.
    - pval_cutoff (float): The p-value cutoff for the likelihood ratio test.

    Returns:
    - list: A list containing the parameters [zero_prob, theta, lambda] of the fitted model.
    - zero_prob (float): Probability of observing zero counts.
    - theta (float): Dispersion parameter (inverse of alpha) for the negative binomial component.
    - lambda (float): Mean parameter for the fitted probability distribution.

    This function fits either a Poisson distribution or a zero-inflated negative binomial model to the count data `x` based on the likelihood ratio test. If the zero-inflated model is preferred, it returns the parameters for that model. Otherwise, it returns the parameters for a Poisson distribution or a negative binomial distribution, depending on the data characteristics.

    Reference: https://www.statsmodels.org/stable/discretemod.html
    """
    mu, var = np.mean(x), np.var(x)
    intercept = np.ones(len(x))
    
    if mu >= var:
        #Poisson
        params = fit_poisson(x, intercept, pval_cutoff)
        params.append('poisson')
        return params
    else:
        #Negative binomial
        mle_nb = sm.NegativeBinomial(x, exog=intercept).fit()
        theta_nb = 1 / mle_nb.params['alpha']
            
        if np.min(x) > 0:
            return [0.0, theta_nb, np.exp(mle_nb.params['const']), "nb"]
        
        else:
            #Zero-inflated negative binomial
            try:
                mle_zinb = sm.ZeroInflatedNegativeBinomialP(x, intercept).fit(method=method_zinb, maxiter=maxiter_zinb, gtol=gtol_zinb)
                theta_zinb = 1 / mle_zinb.params['alpha']
                chisq_val = 2 * (mle_zinb.llf - mle_nb.llf)
                pvalue = 1 - chi2.cdf(chisq_val, 1)
                
                if pvalue < pval_cutoff:
                    return [logistic.cdf(mle_zinb.params['inflate_const']), theta_zinb, np.exp(mle_zinb.params['const']), "zinb"]
                else:
                    return [0.0, theta_nb, np.exp(mle_nb.params['const'], "nb")]
                
            except Exception:
                return [0.0, theta_nb, np.exp(mle_nb.params['const']), "nb"]
            

def fit_marginals(X, marginal='auto', pval_cutoff=0.05):
    """
    Fit a specified model to each species in the given count data.

    Parameters:
    - X (DataFrame): A DataFrame where each row represents a species and each column represents a sample.
    - marginal (str): The model to fit to each species. Options are 'auto', 'zinb', 'nb', and 'poisson'. 
                      'auto' will automatically choose the best model based on the data. Default is 'auto'.
    - pval_cutoff (float): The p-value cutoff for the likelihood ratio test. Only used when marginal='auto'. Default is 0.05.

    Returns:
    - dict: A dictionary containing one key-value pair. The key is 'params' and the value is a 2D numpy array where each row represents a species and each column represents a parameter of the fitted model.

    This function fits a specified model to each species in the count data. The model can be automatically chosen based on the data, or it can be manually specified as zero-inflated negative binomial ('zinb'), negative binomial ('nb'), or Poisson ('poisson').
    """
    p, _ = X.shape
        
    if marginal == 'auto':
        params = np.array([fit_auto(X.iloc[i, :], pval_cutoff=pval_cutoff) for i in range(p)])
    elif marginal == 'zinb':
        params = np.array([fit_zinb(X.iloc[i, :], pval_cutoff=pval_cutoff) for i in range(p)])
    elif marginal == 'nb':
        params = np.array([fit_nb(X.iloc[i, :]) for i in range(p)])
    elif marginal == 'poisson':
        params = np.array([[0.0, np.inf, np.mean(X.iloc[i, :])] for i in range(p)])

    return params