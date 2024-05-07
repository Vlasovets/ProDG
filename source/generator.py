import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2, logistic, norm
from source.utils import uniform_to_marginal, estimate_null_parameters

class DataGenerator:
    def __init__(self, marginal='auto', pval_cutoff=0.05, seed=42):
        self.marginal = marginal
        self.pval_cutoff = pval_cutoff
        self.seed = seed
        self.params = None
        self.models = None

    def fit_poisson(self, X, intercept, pval_cutoff):
        """
        Fit a Poisson distribution or a zero-inflated Poisson model to a given array.

        Parameters:
        - X (array-like): An array of count data.
        - intercept (array-like): An array of intercept values for the model.
        - pval_cutoff (float): The p-value cutoff for the likelihood ratio test.

        Returns:
        - list: A list containing the parameters [zero_prob, theta, lambda] of the fitted model.
        - zero_prob (float): Probability of observing zero counts.
        - theta (float): Dispersion parameter (inverse of alpha) for the Poisson or zero-inflated Poisson model.
        - lambda (float): Mean parameter for the Poisson model.

        This function fits either a Poisson distribution or a zero-inflated Poisson model to the count data `X` based on the likelihood ratio test. If the zero-inflated model is preferred, it returns the parameters for that model. Otherwise, it returns the parameters for a Poisson distribution.

        Reference: https://www.statsmodels.org/stable/discretemod.html
        """

        mle_poisson = sm.GLM(X, intercept, family=sm.families.Poisson()).fit()
        
        try:
            mle_zip = sm.ZeroInflatedPoisson(X, intercept).fit()
            chisq_val = 2 * (mle_zip.llf - mle_poisson.llf)
            pvalue = 1 - chi2.cdf(chisq_val, 1)

            if pvalue < pval_cutoff:
                return [logistic.cdf(mle_zip.params['inflate_const']), np.inf, np.exp(mle_zip.params['const'])]
            else:
                return [0.0, np.inf, np.mean(X)]
            
        except Exception:
            return [0.0, np.inf, np.mean(X)]

    def fit_nb(self, X):
        """
        Fit a negative binomial distribution to a given array.

        Parameters:
        - X (array-like): An array of count data.

        Returns:
        - list: A list containing the parameters [zero_prob, theta, lambda] of the fitted negative binomial model.
        - zero_prob (float): Probability of observing zero counts.
        - theta (float): Dispersion parameter (inverse of alpha) for the negative binomial distribution.
        - lambda (float): Mean parameter for the negative binomial distribution.

        If the data suggests a Poisson distribution is a better fit, it returns the parameters for the Poisson distribution. 
        Otherwise, it returns the parameters for the negative binomial distribution.

        Reference: https://www.statsmodels.org/stable/discretemod.html
        """
        mu, var = np.mean(X), np.var(X)
        intercept = sm.add_constant(np.ones(len(X)))
        
        if mu >= var: 
            # Poisson
            return [0.0, np.inf, mu]
        
        else:
            # Negative binomial
            mle_nb = sm.NegativeBinomial(X, exog=intercept).fit()
            theta_nb = 1 / mle_nb.params['alpha']
                        
            return [0.0, theta_nb, np.exp(mle_nb.params['const'])]

    def fit_zinb(self, X, method='nm', maxiter=5000, gtol=1e-12, pval_cutoff=0.05):
        """
        Fit a zero-inflated negative binomial (ZINB) model or related models to a given count data array.

        Parameters:
        - X (array-like): An array of count data.
        - method (str): The method to use for fitting the model. Default is 'nm'.
        - maxiter (int): The maximum number of iterations. Default is 5000.
        - gtol (float): The gradient tolerance. Default is 1e-12.
        - pval_cutoff (float): The p-value cutoff for the likelihood ratio test. Default is 0.05.

        Returns:
        - list: A list containing the parameters [zero_prob, theta, lambda] of the fitted model, or the result of fit_poisson or fit_nb function.
            - zero_prob (float): Probability of observing zero counts.
            - theta (float): Dispersion parameter (inverse of alpha) for the negative binomial component.
            - lambda (float): Mean parameter for the ZINB model.

        This function fits a ZINB model to the count data `X` if the data suggests it is a better fit. It uses a likelihood ratio test to determine if the zero-inflated model is needed. If a ZINB model is not needed, it falls back to a Poisson or negative binomial model based on the data characteristics.

        Reference: https://www.statsmodels.org/stable/discretemod.html
        """
        
        mu, var = np.mean(X), np.var(X)
        intercept = sm.add_constant(np.ones(len(X)))
        
        if mu >= var:
            return self.fit_poisson(X, intercept, pval_cutoff)
        else:
            if np.min(X) > 0:
                return self.fit_nb(X)
            else:
                try:

                    mle_zinb = sm.ZeroInflatedNegativeBinomialP(X, intercept).fit(method=method, maxiter=maxiter, gtol=gtol)
                    theta_zinb = 1 / mle_zinb.params['alpha']
                        
                    return [logistic.cdf(mle_zinb.params['inflate_const']), theta_zinb, np.exp(mle_zinb.params['const'])]
                
                except Exception:
                    return self.fit_nb(X)

    def fit_auto(self, X, pval_cutoff, method_zinb='nm', maxiter_zinb=5000, gtol_zinb=1e-12):
        """
        Fit parameters for a probability distribution to a given array of count data.

        Parameters:
        - X (array-like): An array of count data.
        - pval_cutoff (float): The p-value cutoff for the likelihood ratio test.

        Returns:
        - list: A list containing the parameters [zero_prob, theta, lambda] of the fitted model.
        - zero_prob (float): Probability of observing zero counts.
        - theta (float): Dispersion parameter (inverse of alpha) for the negative binomial component.
        - lambda (float): Mean parameter for the fitted probability distribution.

        This function fits either a Poisson distribution or a zero-inflated negative binomial model to the count data `X` based on the likelihood ratio test. If the zero-inflated model is preferred, it returns the parameters for that model. Otherwise, it returns the parameters for a Poisson distribution or a negative binomial distribution, depending on the data characteristics.

        Reference: https://www.statsmodels.org/stable/discretemod.html
        """
        mu, var = np.mean(X), np.var(X) # TO DO: switch from empirical mean to the mean of non zeros
        intercept = np.ones(len(X))
        
        if mu >= var: ## TO DO: this creteria is meaningless for microbial data becuase the data is always overdispersed
            #Poisson
            params, model = self.fit_poisson(X, intercept, pval_cutoff), 'poisson'
            return {'params': params, 'model': model}
        else:
            #Negative binomial
            mle_nb = sm.NegativeBinomial(X, exog=intercept).fit()
            theta_nb = 1 / mle_nb.params['alpha']
                
            if np.min(X) > 0:
                params, model = [0.0, theta_nb, np.exp(mle_nb.params['const'])], 'nb'
                return {'params': params, 'model': model}
            
            else:
                #Zero-inflated negative binomial
                try:
                    mle_zinb = sm.ZeroInflatedNegativeBinomialP(X, intercept).fit(method=method_zinb, maxiter=maxiter_zinb, gtol=gtol_zinb)
                    theta_zinb = 1 / mle_zinb.params['alpha']
                    chisq_val = 2 * (mle_zinb.llf - mle_nb.llf)
                    pvalue = 1 - chi2.cdf(chisq_val, 1) # goodness of fit test
                    
                    if pvalue < pval_cutoff:
                        params, model = [logistic.cdf(mle_zinb.params['inflate_const']), theta_zinb, np.exp(mle_zinb.params['const'])], 'zinb'
                        return {'params': params, 'model': model}
                    else:
                        params, model = [0.0, theta_nb, np.exp(mle_nb.params['const'])], 'nb'
                        return {'params': params, 'model': model}
                    
                except Exception:
                    params, model = [0.0, theta_nb, np.exp(mle_nb.params['const'])], 'nb'
                    return  {'params': params, 'model': model}

    def fit_marginals(self, X, marginal='auto', pval_cutoff=0.05):
        """
        Fit a specified model to each species in the given count data.

        Parameters:
        - X (DataFrame): A DataFrame where each row represents a species (p) and each column represents a sample (N).
        - marginal (str): The model to fit to each species. Options are 'auto', 'zinb', 'nb', and 'poisson'. 
                        'auto' will automatically choose the best model based on the data. Default is 'auto'.
        - pval_cutoff (float): The p-value cutoff for the likelihood ratio test. Only used when marginal='auto'. Default is 0.05.

        Returns:
        - dict: A dictionary containing two key-value pairs. The first key is 'params' and the value is a 2D numpy array where each row represents a species and each column represents a parameter of the fitted model. The second key is 'models' and the value is an array where each element represents the model fitted to a species.

        This function fits a specified model to each species in the count data. The model can be automatically chosen based on the data, or it can be manually specified as zero-inflated negative binomial ('zinb'), negative binomial ('nb'), or Poisson ('poisson').
        """
        _, N = X.shape
        result_dict = dict()

        if marginal == 'auto':
            result = np.array([self.fit_auto(X.loc[i], pval_cutoff=pval_cutoff) for i in range(N)])
            params = np.array([item['params'] for item in result])
            models = np.array([item['model'] for item in result])
        elif marginal == 'zinb':
            params = np.array([self.fit_zinb(X.loc[i], pval_cutoff=pval_cutoff) for i in range(N)])
            models = np.full(N, 'zinb')
        elif marginal == 'nb':
            params = np.array([self.fit_nb(X.loc[i]) for i in range(N)])
            models = np.full(N, 'nb')
        elif marginal == 'poisson':
            params = np.array([[0.0, np.inf, np.mean(X.loc[:, i])] for i in range(N)])
            models = np.full(N, 'poisson')

        result_dict['params'] = params
        result_dict['models'] = models

        return result_dict

    def fit(self, X):
        """
        Fit the models to the data.

        Parameters:
        - X (DataFrame): The data to fit the models to.

        This method fits the models to the data `X` using the `fit_marginals` method. The result is stored in `self.marginal`. This method should be called before the `generate` method.
        """
        self.marginal = self.fit_marginals(X)

    def generate_data(self, X, marginal, seed=42):
        """
        Fit a Gaussian copula to the given data.

        Parameters:
        - df (DataFrame): A DataFrame where each row represents a species and each column represents a sample.
        - marginal (dict): A dictionary containing the parameters of the marginal distribution for each species.

        Returns:
        - ndarray: A 2D numpy array where each row represents a species and each column represents a sample. The values are the results of the distribution transformation.

        This function fits a Gaussian copula to the given data based on the parameters of the marginal distribution for each species. It then performs a distribution transformation on the data and returns the results.
        """
        rng = np.random.default_rng(seed)
        p, N = X.shape
        params = estimate_null_parameters(X, marginal, cov_method="scaled", rng=rng)
        sigma = params["R_est"]

        Z = rng.multivariate_normal(mean=np.zeros(p), cov=sigma, size=N)
        Z_cdf = norm.cdf(Z)

        gen_data = uniform_to_marginal(Z_cdf, marginal)

        return gen_data.T

    def generate(self, X):
        """
        Generate new data based on the fitted models.

        Parameters:
        - X (DataFrame): The original data. This is used to get the column names for the generated data.

        Returns:
        - DataFrame: A DataFrame containing the generated data. The column names match those of the original data.

        Raises:
        - Exception: If the `fit` method has not been called before `generate`.
        """
        if self.marginal is None:
            raise Exception("You need to call fit() before generate().")
        generated_data = self.generate_data(X, self.marginal, self.seed)
        
        return pd.DataFrame(generated_data, columns=X.columns)
