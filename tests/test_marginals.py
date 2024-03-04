import unittest
import pandas as pd
import numpy as np
from source.generator import DataGenerator
import statsmodels.api as sm
from scipy.stats import logistic, nbinom


class TestMarginal(unittest.TestCase):
    def setUp(self):
        """
        Set up test data for the test case.

        This method generates a DataFrame with four columns: 'nb', 'poisson', 'zinb', and 'zip'. Each column contains
        100 random numbers generated from a different distribution: negative binomial, Poisson, zero-inflated negative
        binomial, and zero-inflated Poisson, respectively.

        The parameters for the distributions are as follows:
        - Negative binomial: number of successes = 10, probability of success = 0.5
        - Poisson: lambda = 5
        - Zero-inflated negative binomial: same as negative binomial, but with a 50% chance of being replaced with zero
        - Zero-inflated Poisson: same as Poisson, but with a 50% chance of being replaced with zero

        The generated DataFrame is stored in the instance variable `self.df` for use in the test methods.
        """
        size = 1000  # size of the data
        n = 10  # number of successes for negative binomial
        p = 0.5  # probability of success for negative binomial and Poisson
        lambda_ = 5  # lambda parameter for Poisson
        zero_prob = 0.5  # probability of zero for zero-inflated distributions

        nb_data = np.random.negative_binomial(n, p, size=size)
        poisson_data = np.random.poisson(lambda_, size=size)
        zinb_data = nbinom.rvs(n, p, size=size)
        zinb_data[np.random.random(size) < zero_prob] = 0
        zip_data = np.random.poisson(lambda_, size=size)
        zip_data[np.random.random(size) < zero_prob] = 0

        self.df = pd.DataFrame({
            'nb': nb_data,
            'poisson': poisson_data,
            'zinb': zinb_data,
            'zip': zip_data
        })

    def test_nb(self):
        """
        Test the fit_nb function with negative binomial data.

        This test checks that the fit_nb function correctly estimates the parameters of a negative binomial distribution.
        The test generates a negative binomial model using the statsmodels library, then compares the parameters estimated
        by statsmodels with the parameters estimated by fit_nb.

        The parameters being tested are theta (1 / alpha) and mu (exp(const)).

        The test uses the assertAlmostEqual method to check that the estimated parameters are close to the true parameters,
        up to a certain number of decimal places.
        """
        x = self.df['nb']
        intercept = sm.add_constant(np.ones(len(x)))

        mle_nb = sm.NegativeBinomial(x, exog=intercept).fit()

        theta_nb= 1 / mle_nb.params['alpha']
        mu = np.exp(mle_nb.params['const'])


        generator = DataGenerator()
        _, test_theta_nb, test_mu = generator.fit_nb(x)

        self.assertAlmostEqual(test_theta_nb, theta_nb)
        self.assertAlmostEqual(test_mu, mu)

    def test_poisson(self):
        """
        Test the fit_poisson function with zero-inflated Poisson data.

        This test checks that the fit_poisson function correctly estimates the parameters of a zero-inflated Poisson distribution.
        The test generates a zero-inflated Poisson model using the statsmodels library, then compares the parameters estimated
        by statsmodels with the parameters estimated by fit_poisson.

        The parameters being tested are zero_prob (probability of observing zero counts) and mu (mean parameter for the Poisson model).

        The test uses the assertAlmostEqual method to check that the estimated parameters are close to the true parameters,
        up to a certain number of decimal places.

        The fit_poisson function is called with a p-value cutoff of 0.05, meaning that the zero-inflated model will be chosen
        if the p-value of the likelihood ratio test is less than 0.05.
        """
        x = self.df['zip']
        intercept = sm.add_constant(np.ones(len(x)))

        mle_zip = sm.ZeroInflatedPoisson(x, intercept).fit()

        zero_prob = logistic.cdf(mle_zip.params['inflate_const'])
        mu = np.exp(mle_zip.params['const'])

        generator = DataGenerator()
        test_zero_prob, _, test_mu = generator.fit_poisson(x, intercept=intercept, pval_cutoff=0.05)

        self.assertAlmostEqual(test_zero_prob, zero_prob)
        self.assertAlmostEqual(test_mu, mu)

    def test_zinb(self):
        """
        This test checks that the fit_zinb function correctly estimates the parameters of a zero-inflated Negative Binomial (NB) distribution.
        The test generates a zero-inflated NB model using the statsmodels library, then compares the parameters estimated
        by statsmodels with the parameters estimated by fit_zinb.
        """
        x = self.df['zinb']
        intercept = sm.add_constant(np.ones(len(x)))

        mle_zinb = sm.ZeroInflatedNegativeBinomialP(x, intercept).fit(method='nm', maxiter=5000, gtol=1e-12)

        zero_prob = logistic.cdf(mle_zinb.params['inflate_const'])
        theta_zinb= 1 / mle_zinb.params['alpha']
        mu = np.exp(mle_zinb.params['const'])

        generator = DataGenerator()
        test_zero_prob, test_theta_zinb, test_mu = generator.fit_zinb(x)

        self.assertAlmostEqual(test_zero_prob, zero_prob)
        self.assertAlmostEqual(test_theta_zinb, theta_zinb)
        self.assertAlmostEqual(test_mu, mu)

    def test_fit_marginals_auto(self):
        """
        Test the fit_marginals function with 'auto' as the marginal parameter.

        This test checks that the fit_marginals function correctly assigns inf values 
        only to rows where the model is 'poisson'. It does this by calling fit_marginals 
        with 'auto' as the marginal parameter, converting the returned parameters to a 
        DataFrame, and checking for any inf values in rows where the model is not 'poisson'. 
        If there are any such values, the test fails.
        """
        generator = DataGenerator()
        marginal = generator.fit_marginals(self.df, marginal='auto', pval_cutoff=0.05)

        marginal_df = pd.DataFrame(data=marginal['params'], columns=['zero_prob', 'theta', 'lambda'])
        marginal_df['model'] = marginal['models']

        mask_model_not_poisson = marginal_df['model'] != 'poisson'
        mask_inf_values = np.isinf(marginal_df[['zero_prob', 'theta', 'lambda']]).any(axis=1)

        non_poisson_rows_with_inf = marginal_df[mask_model_not_poisson & mask_inf_values]

        self.assertTrue(non_poisson_rows_with_inf.empty)


if __name__ == '__main__':
    unittest.main()