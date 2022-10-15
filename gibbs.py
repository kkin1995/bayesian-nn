import numpy as np

class GibbsSampling:
    def __init__(self, mu_1, sigma_1, mu_2, sigma_2, rho):
        self.mu_1 = mu_1
        self.sigma_1 = sigma_1
        self.mu_2 = mu_2
        self.sigma_2 = sigma_2
        self.rho = rho
        self.x_1 = np.linspace(-10, 10, 100)
        self.x_2 = np.linspace(-10, 10, 100)
        self.X_1, self.X_2 = np.meshgrid(self.x_1, self.x_2)

    def __bivariate_normal(self):
        factor = (1 / (2*np.pi))
        exponential_factor = -(1/2)*( (1)/(1-self.rho**2)*( ((self.X_1 - self.mu_1)/(self.sigma_1))**2 + ((self.X_2 - self.mu_2)/(self.sigma_2))**2 - 2*self.rho*((self.X_1 - self.mu_1)/(self.sigma_1))*((self.X_2 - self.mu_2)/(self.sigma_2)) ) )
        exponential = np.exp(exponential_factor)
        return factor * exponential

    def __conditional(self, X, Y_const):
        factor = 1 / (np.sqrt(2*np.pi) * self.sigma_1 * np.sqrt(1-(self.rho**2)))
        exponential_factor = ( -((1)/(2 * (self.sigma_1**2) * (1 - (self.rho**2)))) * ( X - ( self.mu_1 + self.sigma_1 * self.rho * ( (Y_const - self.mu_2)/(self.sigma_2) ) ) )**2 )
        exponential = np.exp(exponential_factor)
        return factor * exponential

    def run(self):
        self.bi_gauss = self.__bivariate_normal()
        self.bi_gauss /= self.bi_gauss.sum()

        x_1_samples = []
        x_2_samples = []

        x_1_samples.append(np.random.choice(self.X_1.ravel(), replace = False, p = self.bi_gauss.ravel()))
        x_2_samples.append(np.random.choice(self.X_1.ravel(), replace = False, p = self.bi_gauss.ravel()))

        T = 10000
        for i in range(1, T):
            p_x1_fix_x2 = self.__conditional(self.X_1.ravel(), x_2_samples[i-1])
            p_x1_fix_x2 /= p_x1_fix_x2.sum()
            x_1_samples.append(np.random.choice(self.X_1.ravel(), replace = False, p = p_x1_fix_x2.ravel()))
            p_x2_fix_x1 = self.__conditional(self.X_2.ravel(), x_1_samples[i])
            p_x2_fix_x1 /= p_x2_fix_x1.sum()
            x_2_samples.append(np.random.choice(self.X_2.ravel(), replace = False, p = p_x2_fix_x1.ravel()))

        return x_1_samples, x_2_samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    mu_1 = 0
    sigma_1 = 1
    mu_2 = 1
    sigma_2 = 1
    rho = 0.1

    gibbs_sampler = GibbsSampling(mu_1, sigma_1, mu_2, sigma_2, rho)
    x1_samples, x2_samples = gibbs_sampler.run()

    fig, axs = plt.subplots(2,1)
    axs[0].contourf(gibbs_sampler.X_1, gibbs_sampler.X_2, gibbs_sampler.bi_gauss)
    axs[1].hist2d(x1_samples, x2_samples, bins = 100, range = [[-10, 10], [-10, 10]])
    plt.show()