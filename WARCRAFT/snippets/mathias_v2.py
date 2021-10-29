#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt


def sample_gamma(batch_size: int,
                 nb_samples: int,
                 k: int,
                 nb_iterations: int = 1000) -> np.ndarray:
    samples = np.zeros((batch_size, nb_samples, k), dtype='float')
    for i in range(1, nb_iterations + 1):
        gs = np.random.gamma(1.0 / k, k / i, size=(batch_size, nb_samples, k))
        samples = samples + gs
    samples = samples - np.log(nb_iterations)
    samples = samples / k
    res = np.sum(samples, axis=2)
    return res

samples = sample_gamma(batch_size=8, nb_samples=1000, k=20)

print('XXX', samples.shape)

count, bins, ignored = plt.hist(samples[1, :], 20, density=True)

mu, beta = 0.0, 1.0
y = (1 / beta) * np.exp(-(bins - mu) / beta) * np.exp(-np.exp(-(bins - mu) / beta))

plt.plot(bins, y, linewidth=2, color='r')
plt.show()
