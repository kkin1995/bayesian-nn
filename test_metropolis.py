import numpy as np
import matplotlib.pyplot as plt
from metropolis import MetropolisHasting

class BiModalGaussian:
        def __init__(self, mu_1, sigma_1, mu_2, sigma_2, A_1 = 1, A_2 = 1):
            self.A_1 = A_1
            self.mu_1 = mu_1
            self.sigma_1 = sigma_1
            self.A_2 = A_2
            self.mu_2 = mu_2
            self.sigma_2 = sigma_2

        def __normal(self, x, mu, sigma):
            numerator = np.exp((-(x - mu)**2)/(2 * sigma**2))
            denominator = sigma = np.sqrt(2 * np.pi)
            return numerator / denominator
    
        def distribution(self, size = 1):
            xs = np.linspace(-10, 10, size)
            gauss_1 = self.A_1 * np.array([self.__normal(x, self.mu_1, self.sigma_1) for x in xs])
            gauss_2 = self.A_2 * np.array([self.__normal(x, self.mu_2, self.sigma_2) for x in xs])
            return np.concatenate((gauss_1, gauss_2))

        def sample(self, x):
            gauss_1 = self.A_1 * self.__normal(x, self.mu_1, self.sigma_1)
            gauss_2 = self.A_2 * self.__normal(x, self.mu_2, self.sigma_2)
            return gauss_1 + gauss_2

class Gaussian:
    def __init__(self, mu, sigma, A = 1):
        self.mu = mu
        self.sigma = sigma
        self.A = A

    def __normal(self, x, mu, sigma):
        numerator = np.exp((-(x - mu)**2)/(2 * sigma**2))
        denominator = sigma = np.sqrt(2 * np.pi)
        return numerator / denominator
    
    def distribution(self, size = 1):
        xs = np.linspace(-10, 10, size)
        gauss = self.A * np.array([self.__normal(x, self.mu, self.sigma) for x in xs])
        return gauss
    
    def sample(self, x):
        gauss = self.A * self.__normal(x, self.mu, self.sigma)
        return gauss


if __name__ == "__main__":
    POSTERIOR_TO_SIMULATE = "GAUSSIAN"

    xs = np.linspace(-10, 10, 100)
    initlalSample = 1
    n_samples = 10000
    step_size = 1

    if POSTERIOR_TO_SIMULATE == "BI MODAL GAUSSIAN":

        mu_1 = 2
        sigma_1 = 1
        mu_2 = -1
        sigma_2 = 1
        
        bi_modal = BiModalGaussian(mu_1, sigma_1, mu_2, sigma_2)
        bi_modal_dist = bi_modal.distribution(size = 100)
        metropolis = MetropolisHasting(initlalSample, n_samples, step_size, bi_modal.sample)
        posterior = metropolis.run()

        plt.plot(xs, bi_modal_dist)
        plt.hist(posterior, bins = 50, density=True)

        plt.show()

    elif POSTERIOR_TO_SIMULATE == "GAUSSIAN":
        
        mu = 0
        sigma = 1

        gaussian = Gaussian(mu, sigma)
        gaussian_dist = gaussian.distribution(size = 100)
        metropolis = MetropolisHasting(initlalSample, n_samples, step_size, gaussian.sample)
        posterior = metropolis.run()

        plt.plot(xs, gaussian_dist)
        plt.hist(posterior, bins = 50, density=True)

        plt.show()