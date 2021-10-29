#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import multiprocessing

import numpy as np

import torch
from torch.distributions import *

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
torch.set_num_threads(multiprocessing.cpu_count())


gamma_cache = {}
gumbel_cache = {}


def sample_gamma(nb_samples: int, nb_iterations: int, k: int, device: torch.device) -> torch.Tensor:
    global gamma_cache
    res = torch.zeros(nb_samples).to(device)
    for i in range(1, nb_iterations + 1):
        key = (k, i)
        if key not in gamma_cache:
            gamma_cache[key] = Gamma(1.0 / k, k / i)
        gamma = gamma_cache[key]
        res = res + gamma.sample(sample_shape=[nb_samples]).to(device)
    res = (res - np.log(nb_iterations)) / k
    return res


def sample_gumbel(nb_samples: int, device: torch.device) -> torch.Tensor:
    global gumbel_cache
    key = (0, 1)
    if key not in gumbel_cache:
        gumbel_cache[key] = Gumbel(torch.tensor([key[0]], dtype=torch.float64), torch.tensor([key[1]], dtype=torch.float64))
    gumbel = gumbel_cache[key]
    res = gumbel.sample(sample_shape=[nb_samples]).to(device)
    return res


def KL(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.sum(p * torch.log(p / q))
    # return F.kl_div(torch.log(p).reshape(1, -1), q.reshape(1, -1), reduction='sum')


def sym_KL(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (KL(p, q) + KL(q, p)) * 0.5


def set_seed(seed: int, is_deterministic: bool = True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return


def main(argv):
    set_seed(0)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device('cuda')

    uniform = Uniform(low=-5, high=5)
    weights = uniform.sample(sample_shape=torch.tensor([4, 6])).to(device)

    nb_weights = weights.shape[0] * weights.shape[1]
    c_logits = weights.sum(1)

    categorical = OneHotCategorical(logits=c_logits)
    # s = categorical.sample().to(device)

    logits = categorical.logits
    probs = categorical.probs

    print('Probabilities', probs)

    eye = torch.eye(weights.shape[0]).to(device)
    map_ = eye[torch.argmax(logits)]

    print('MAP state', map_)
    
    nb_samples = 1000

    for t in [1e-1, 1e0]:
        for m in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            eps = sample_gamma(nb_samples=nb_weights * nb_samples, nb_iterations=50, k=weights.shape[1], device=device)
            eps = eps.reshape(-1, weights.shape[0], weights.shape[1])

            samples = weights + (eps * t * m)
            marginals = eye[samples.sum(2).max(1).indices].mean(0)
            div = sym_KL(probs, marginals)

            print(f'Gamma Marginals (𝜏 = {t * m:.5f})\tKL: {div}')

    for t in [1e-2, 1e-1, 1e0]:
        for m in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            eps = sample_gumbel(nb_samples=nb_weights * nb_samples, device=device)
            eps = eps.reshape(-1, weights.shape[0], weights.shape[1])

            samples = weights + (eps * t * m)
            marginals = eye[samples.sum(2).max(1).indices].mean(0)
            div = sym_KL(probs, marginals)

            print(f'Gumbel Marginals (𝜏 = {t * m:.5f})\tKL: {div}')



if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    with torch.no_grad():
        main(sys.argv[1:])
