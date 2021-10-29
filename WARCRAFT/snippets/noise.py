#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np

from torch.distributions import Gumbel

import matplotlib.pyplot as plt

import torch


def main():
    m = Gumbel(torch.tensor([0.0]), torch.tensor([1.0]))
    samples = m.sample(sample_shape=torch.Size([10000]))

    print('XXX', samples.shape)

    samples = samples.view(-1).detach().cpu().numpy()

    samples = samples / float(sys.argv[1])

    # mu, sigma = 100, 15
    # samples = mu + sigma * np.random.randn(10000)

    n, bins, patches = plt.hist(samples,
                                bins=50,
                                density=True)

    plt.xlabel('Samples')
    plt.ylabel('Probability')
    plt.title('Histogram of Gumbel')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
