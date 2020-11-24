import numpy as np
from scipy.stats import norm
for i in range(10):
    seed = i
    np.random.seed(seed=seed)
    size = 10
    U = np.random.randint(30, 200, size=size)

    D = np.random.randint(1000, 60000, size=size)

    ot1 = np.random.randint(0, 120)

    def sigma(x):
        return np.log((np.dot(D, x) + np.dot(U, x) ** 2) / (np.dot(U, x) ** 2))

    def log_normal_cc(x):
        return np.log(np.dot(U, x)) - 1 / 2 * sigma(x) + norm.ppf(0.95) * np.sqrt(sigma(x))

    x1 = np.random.randint(0, 2, size=size)

    def grad(x):
        return 2 * U / (np.dot(U, x)) - 1 / 2 * (D + 2 * U * np.dot(U, x)) / (np.dot(D, x) + np.dot(U, x) ** 2) + norm.ppf(0.95) / 2 * sigma(x) ** (-0.5) * ((np.dot(U, x) * D - 2 * np.dot(D, x) * U) / ((np.dot(D, x) + np.dot(U, x) ** 2) * np.dot(U, x)))

    a = np.random.randint(0, 2, size=size)
    ot_2 = np.random.randint(0, 120)

    aprox = log_normal_cc(a) + np.dot(grad(a), x1 - a)

    print('線形近似={:}'.format(aprox - np.log(ot1 + 480)))
    print('Truth={:}'.format(log_normal_cc(x1) - np.log(ot1 + 480)))
    print()
