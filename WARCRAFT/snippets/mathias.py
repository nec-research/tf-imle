import math
import numpy as np

import matplotlib.pyplot as plt

mu = 0
beta = 1

# expected number of 1s in the solution (=k in our discussion)
k = 20
# the number of iterations to approximate limit of m->infty
approx_iters = 10
# def GumbelKSum(k, approx_iters = 100, nb_samples=1)
size = 1000
# the samples we generate
s = np.zeros((size, 1, k), dtype='float')
# here we assume k is number of 1s, so we want 1 sample per perturbation
# t = np.zeros((nb_samples, 1, k))
# these are the number of iterations to approximate the limit sum
# the following is the formula that allows us to express a Gumbel as a finite sum
for i in range(1, approx_iters + 1):
    gs = np.random.gamma(1.0 / float(k), float(k) / float(i), size=(size, 1, k))
    s = s + gs

# the log(m) term
s = s - math.log(approx_iters)
# divide by k --> each s[c] has k samples whose sum is distributed as Gumbel(0, 1)
s = s / float(k)

# turn into an array
print(s.shape)
# sum the k perturbations
s_final = np.sum(s, axis=2)
print(s_final.shape)
count, bins, ignored = plt.hist(s_final, 30, density=True)
plt.plot(bins, (1 / beta) * np.exp(-(bins - mu) / beta)
         * np.exp(-np.exp(-(bins - mu) / beta)),
         linewidth=2, color='r')
plt.show()
