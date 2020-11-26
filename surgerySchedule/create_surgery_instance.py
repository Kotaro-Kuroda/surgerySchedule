import numpy as np

def create_surgery(mean, variance):
    sigma = np.log(variance / (mean ** 2) + 1)
    mu = np.log(mean) - 1 / 2 * sigma
    return np.random.lognormal(mu, sigma)
