"""Straight through estimator"""

import torch


def ste(sampler):
    return lambda theta: _StraightThroughEstimator.apply(theta, sampler)


# noinspection PyMethodOverriding
class _StraightThroughEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, sampler):
        return sampler(theta)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None
