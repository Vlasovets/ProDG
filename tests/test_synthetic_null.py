import unittest
import pandas as pd
import numpy as np
from source.generator import DataGenerator
from scipy.stats import nbinom


class TestSyntheticNull(unittest.TestCase):
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

    def test_gaussian_copula(self):
        generator = DataGenerator()
        marginal = generator.fit_marginals(self.df, marginal='auto')
        gen_data = generator.generate_data(self.df, marginal)

        oracle_R = np.corrcoef(self.df, rowvar=False)
        gen_R = np.corrcoef(gen_data, rowvar=False)

        self.assertTrue(np.allclose(oracle_R, gen_R, atol=0.1), "The correlation matrices of the original and synthetic data are not close enough.")


if __name__ == '__main__':
    unittest.main()