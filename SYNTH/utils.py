import inspect

import torch
import numpy as np


def expect_obj(dist, theta, obj):
    """
    Computes \mathbb{E}_{z\sim dist(z, theta)} [ obj(z) ] =
                            = sum_{z in dist.states} dist(z) * obj(z)

    :param dist:
    :param theta:
    :param obj:
    :return:
    """
    _pmf = dist.pmf(theta)
    states = dist.states
    _p_values = torch.stack([p_z * obj(z) for p_z, z in zip(_pmf, states)])
    return torch.sum(_p_values)


def sum_of_gamma_noise(k_gamma, tau=1., rng=None, s=10):
    if rng is None: rng = np.random.RandomState()
    return lambda: torch.tensor((tau / k_gamma) * (
            np.sum(
                [rng.gamma(1.0 / k_gamma, k_gamma / (i + 1.0)) for i in range(s)])
            - np.log(s)
    )).float()


def gumbel_noise(tau=1., rng=None):
    if rng is None: rng = np.random.RandomState()
    return lambda: torch.tensor(rng.gumbel(0, tau)).float()


if __name__ == '__main__':
    sog_f = sum_of_gamma_noise(3., 1.)
    print(sog_f())
    print(sog_f())

    ppp = torch.stack([sog_f() for _ in range(3)])

    print(ppp)


def _maybe_ctx_call(func, ctx, theta):
    args = inspect.getfullargspec(func).args
    if 'ctx' in args:
        return func(theta, ctx=ctx)
    else:
        return func(theta)