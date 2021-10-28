"""Implicit maximum likelihood estimator (I-MLE)"""

import torch

from SYNTH.utils import _maybe_ctx_call


def imle_pid(lmd, sampler, use_fw_pass_for_mu_p=True, marginals_approx=None,
             normalized=False):
    """
    I-MLE ``layer'' with target distribution given by perturbation-based implicit differentiation.

    :param lmd: perturbation intensity (scalar)
    :param sampler: method to sample from the distribution (e.g. sample, perturb and MAP or MAP)
                    can also be used to sample multiple times
    :param use_fw_pass_for_mu_p: if True (default) use the forward pass to approximate the marginals of p
    :param marginals_approx: (optional) function to approximate the marginals, default is (one sample using) `sampler`
    :param normalized: if true, the perturbation intensity is proportional to the norm of theta (default=False)
                        this needs to be thought over a bit more...

    :return: a function (`torch.autograd.Function`) with forward and backward fully implemented!
    """
    return lambda theta: _IMLE_PID.apply(
        theta, lmd, sampler, use_fw_pass_for_mu_p, marginals_approx, normalized
    )


# noinspection PyMethodOverriding
class _IMLE_PID(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, lmd, sample_strategy,
                use_fw_pass_for_mu_p=True, marginals_approx=None,
                normalized=False):
        # save stuff for backward pass
        ctx.ttheta = theta
        ctx.normalized = normalized
        ctx.use_fw_pass_for_mu_p = use_fw_pass_for_mu_p
        ctx.lmd = lmd
        ctx.sample_strategy = sample_strategy
        ctx.marginal_approx = marginals_approx if marginals_approx is not None else sample_strategy

        # actual forward
        ctx.fw = _maybe_ctx_call(sample_strategy, ctx, theta)
        return ctx.fw

    @staticmethod
    def backward(ctx, grad_outputs):
        norm_theta = torch.norm(ctx.ttheta) if ctx.normalized else 1.
        # perturbation-based q
        mu_p = ctx.fw if ctx.use_fw_pass_for_mu_p else _maybe_ctx_call(ctx.marginal_approx, ctx, ctx.ttheta)
        theta_prime = ctx.ttheta - ctx.lmd * norm_theta * grad_outputs
        mu_q = _maybe_ctx_call(ctx.marginal_approx, ctx, theta_prime)
        return mu_p - mu_q, None, None, None, None, None
