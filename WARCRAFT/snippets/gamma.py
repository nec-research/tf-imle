import numpy as np
import matplotlib.pyplot as plt

from typing import Union


def sample_sum_of_gammas(nb_samples: int,
                         k: np.ndarray,
                         nb_iterations: int = 10,
                         temperature: Union[int, np.ndarray] = 1.):
    """These should be the epsilons from Th, 1, """
    k_size = k.shape[0]
    samples = np.zeros((nb_samples, k_size), dtype='float')
    for i in range(1, nb_iterations + 1):
        gs = np.random.gamma(1. / k, k / i, size=[nb_samples, k_size])
        samples = samples + gs
    res = temperature * ((samples - np.log(nb_iterations)) / k)
    # res = (np.sum([np.random.gamma(1. / k, k / i, size=size) for i in range(1, s + 1)], axis=0) - np.log(s)) / k
    return res


nb_samples = 10080 * 100

k = [12, 6, 12, 12, 12, 12, 14]
k_np = np.array(k, dtype=np.float)

nb_iterations = 100

tmp = sample_sum_of_gammas(nb_samples, k_np, nb_iterations)

print(tmp.shape)

sum_of_tmps = []

for j, k_value in enumerate(k):
    sum_of_tmp_entry = []
    count = 0
    while count < nb_samples:
        _tmp_sum_sum = np.sum(tmp[count:count + k_value, j])  # this should be ~ Gamma(0, 1)
        sum_of_tmp_entry += [_tmp_sum_sum]
        count += k_value
    sum_of_tmps += [sum_of_tmp_entry]
    print(k_value, len(sum_of_tmp_entry))  # number of samples (sums) per column

for cc in sum_of_tmps:
    count, bins, ignored = plt.hist(cc, 50, density=True)
    mu, beta = 0.0, 1.0
    y = (1 / beta) * np.exp(-(bins - mu) / beta) * np.exp(-np.exp(-(bins - mu) / beta))
    plt.plot(bins, y, linewidth=2, color='r')
    plt.show()
