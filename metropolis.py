import numpy as np

class MetropolisHasting:
    """
    Creates a single instance of the Metropolis Hasting algorithm

    Attributes
    -----------
    initialSample: float
        first element in the markov chain
    n_samples: int
        number of samples to compute in the markov chain
    step_size: float
        standard deviation of gaussian at each candidate
    proposed_dist: instance
        method of the instance that samples single points from the proposed
    """
    def __init__(self, initialSample, n_samples, step_size, proposed_dist):
        self.initialSample = initialSample
        self.n_samples = n_samples
        self.proposed_dist = proposed_dist
        self.step_size = step_size

    def __normal(self, x, mu, sigma):
        numerator = np.exp((-(x - mu)**2)/(2 * sigma**2))
        denominator = sigma = np.sqrt(2 * np.pi)
        return numerator / denominator
    
    def run(self):
        burn_in = int(self.n_samples * 0.2)
        self.chain = []
        self.chain.append(self.initialSample)
        sample_idx = 0
        while sample_idx < self.n_samples:
            dist_at_sample = np.random.normal(self.chain[sample_idx], self.step_size, size = 100)
            sample = np.random.choice(dist_at_sample)
            alpha = self.proposed_dist(sample) / self.proposed_dist(self.chain[sample_idx])
            u = np.random.uniform(0, 1)
            if alpha > u:
                self.chain.append(sample)
            else:
                self.chain.append(self.chain[sample_idx])
            sample_idx += 1
        return self.chain[burn_in:]