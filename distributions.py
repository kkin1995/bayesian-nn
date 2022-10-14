import numpy as np
import matplotlib.pyplot as plt
from metropolis import MetropolisHasting

class Distributions:
    def __init__(self):
        pass

    def normal(self, x, mu, sigma):
        numerator = np.exp((-(x - mu)**2)/(2 * sigma**2))
        denominator = sigma = np.sqrt(2 * np.pi)
        return numerator / denominator

    def bivariate_normal(self, X_1, X_2, mu_1, sigma_1, mu_2, sigma_2, rho):
        factor = (1 / (2*np.pi))
        exponential_factor = -(1/2)*( (1)/(1-rho**2)*( ((X_1 - mu_1)/(sigma_1))**2 + ((X_2 - mu_2)/(sigma_2))**2 - 2*rho*((X_1 - mu_1)/(sigma_1))*((X_2 - mu_2)/(sigma_2)) ) )
        exponential = np.exp(exponential_factor)
        return factor * exponential

class BiModalGaussian(Distributions):
        def __init__(self, mu_1, sigma_1, mu_2, sigma_2, A_1 = 1, A_2 = 1):
            super(BiModalGaussian, self).__init__()
            
            self.A_1 = A_1
            self.mu_1 = mu_1
            self.sigma_1 = sigma_1
            self.A_2 = A_2
            self.mu_2 = mu_2
            self.sigma_2 = sigma_2
    
        def distribution(self, size = 1):
            xs = np.linspace(-10, 10, size)
            gauss_1 = self.A_1 * np.array([self.normal(x, self.mu_1, self.sigma_1) for x in xs])
            gauss_2 = self.A_2 * np.array([self.normal(x, self.mu_2, self.sigma_2) for x in xs])
            return np.concatenate((gauss_1, gauss_2))

        def sample(self, x):
            gauss_1 = self.A_1 * self.normal(x, self.mu_1, self.sigma_1)
            gauss_2 = self.A_2 * self.normal(x, self.mu_2, self.sigma_2)
            return gauss_1 + gauss_2

class Gaussian(Distributions):
    def __init__(self, mu, sigma, A = 1):
        super(Gaussian, self).__init__()

        self.mu = mu
        self.sigma = sigma
        self.A = A
    
    def distribution(self, size = 1):
        xs = np.linspace(-10, 10, size)
        gauss = self.A * np.array([self.normal(x, self.mu, self.sigma) for x in xs])
        return gauss
    
    def sample(self, x):
        gauss = self.A * self.normal(x, self.mu, self.sigma)
        return gauss

class BiVariateGaussian(Distributions):
    def __init__(self, mu_1, mu_2, sigma_1, sigma_2, rho):
        super(BiVariateGaussian, self).__init__()
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.rho = rho

    def distribution(self, size = 1):
        X_1 = np.linspace(-10, 10, size)
        X_2 = np.linspace(-10, 10, size)
        gauss = self.bivariate_normal(X_1, X_2, self.mu_1, self.sigma_1, self.mu_2, self.sigma_2, self.rho)
        return gauss

    def conditional(self, X, Y_const, fix = "X2"):
        if fix == "X1":
            X = self.X_2
        else:
            X = self.X_1
        factor = 1 / (np.sqrt(2*np.pi) * self.sigma_1 * np.sqrt(1-(self.rho**2)))
        exponential_factor = ( -((1)/(2 * (self.sigma_1**2) * (1 - (self.rho**2)))) * ( X - ( self.mu_1 + self.sigma_1 * self.rho * ( (Y_const - self.mu_2)/(self.sigma_2) ) ) )**2 )
        exponential = np.exp(exponential_factor)
        return factor * exponential

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