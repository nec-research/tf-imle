#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

rs = np.random.RandomState(0)

# We want to minimise this function but we can't use its analytic gradient
def f(x: float) -> float:
    y = 10 + 20 * x + 1.5 * x ** 2
    # f is stochastic
    y += rs.normal(0) * 10
    return y


mu = 0.0
nb_trials = 1000

for i in range(10000):
    # Estimate the gradient of E_p[f(z)] wrt mu, where p(z;mu) = N(mu, I)
    gradient = 0.0
    for j in range(nb_trials):
        # Sample z from N(mu, I)
        z = rs.normal(mu)
        # f(z) * gradient of the Gaussian density's log-likelihood wrt concentration parameter mu
        # Why? Here's the derivation: http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/
        gradient += (f(z) * (z - mu)) / nb_trials

    # Update mu in the (estimated) steepest descent direction
    mu = mu - 0.001 * gradient

    print(f'{i}\tmu: {mu:.3f}, f(mu): {f(mu):.3f}')
