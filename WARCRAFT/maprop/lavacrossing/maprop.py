# -*- coding: utf-8 -*-

import numpy as np
import torch

from maprop.blackbox.losses import HammingLoss

from maprop.lavacrossing.trainers import ShortestPathAbstractTrainer

from maprop.blackbox.dijkstra import get_solver
from maprop.utils import maybe_parallelize

from maprop.models import get_model

from torch.distributions import Gumbel

from typing import Dict, Any


class ShortestPathMAP(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                weights,
                lambda_val,
                mode: Dict[str, Any],
                neighbourhood_fn="8-grid"):
        ctx.lambda_val = lambda_val
        ctx.neighbourhood_fn = neighbourhood_fn
        ctx.solver = get_solver(neighbourhood_fn)

        ctx.mode = mode

        ctx.weights = weights.detach().cpu().numpy()
        ctx.suggested_tours = np.asarray(maybe_parallelize(ctx.solver, arg_list=list(ctx.weights)))
        return torch.from_numpy(ctx.suggested_tours).float().to(weights.device)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.shape == ctx.suggested_tours.shape
        grad_output_numpy = grad_output.detach().cpu().numpy()

        if ctx.mode.type == 0:
            weights_prime = np.maximum(ctx.weights + ctx.lambda_val * grad_output_numpy, 0.0)
        elif ctx.mode.type == 1:
            weights_prime = np.maximum(ctx.lambda_val * grad_output_numpy, 0.0)
        elif ctx.mode.type == 2:
            weights_prime = np.maximum(grad_output_numpy, 0.0)
        elif ctx.mode.type == 3:
            weights_prime = np.maximum(- grad_output_numpy, 0.0)
        else:
            raise ValueError(f'Unknown mode type {ctx.mode}')

        better_paths = np.asarray(maybe_parallelize(ctx.solver, arg_list=list(weights_prime)))

        gradient = - (ctx.suggested_tours - better_paths) / ctx.lambda_val

        if 'use_marginal' in ctx.mode and ctx.mode.use_marginal is True:
            orig_weights = ctx.weights

            if 'nb_samples' not in ctx.mode:
                raise ValueError('Number of samples to be used to compute the marginal (mode.nb_samples) unknown')
            nb_samples = ctx.mode.nb_samples
            temperature = ctx.mode.temperature if 'temperature' in ctx.mode else 1.0

            # [SAMPLES, BATCH_SIZE, HEIGHT, WIDTH]
            eps_shape = [nb_samples, orig_weights.shape[0], orig_weights.shape[1], orig_weights.shape[2]]

            with torch.no_grad():
                gumbel = Gumbel(torch.tensor([0.0]), torch.tensor([1.0]))

                eps = gumbel.sample(sample_shape=torch.Size(eps_shape))
                eps = eps.reshape(eps_shape).detach().cpu().numpy() / temperature

            # [SAMPLES, BATCH_SIZE, HEIGHT, WIDTH] + [BATCH_SIZE, HEIGHT, WIDTH]
            # will have shape [SAMPLES, BATCH_SIZE, HEIGHT, WIDTH] thanks to broadcasting
            sample_weights = orig_weights + eps
            sample_weights = sample_weights.reshape([-1, orig_weights.shape[1], orig_weights.shape[2]])

            # We solve [SAMPLES, BATCH_SIZE, HEIGHT, WIDTH] instances of the shortest paths problem
            sample_tours = np.asarray(maybe_parallelize(ctx.solver, arg_list=list(sample_weights)))
            sample_tours = sample_tours.reshape(eps_shape)

            # We compute the mean on the SAMPLE dimension (marginal)
            marginals = np.mean(sample_tours, axis=0)

            gradient = - (marginals - better_paths) / ctx.lambda_val

        return torch.from_numpy(gradient).to(grad_output.device), None, None, None


class DijkstraMAP(ShortestPathAbstractTrainer):
    def __init__(self, *, l1_regconst, lambda_val,
                 mode,
                 **kwargs):
        super().__init__(**kwargs)
        self.l1_regconst = l1_regconst
        self.lambda_val = lambda_val

        self.mode = mode
        print(f'MAP-BACKPROP MODE: {self.mode}')

        # self.solver = ShortestPath(lambda_val=lambda_val, neighbourhood_fn=self.neighbourhood_fn)
        self.sp_fun = ShortestPathMAP.apply

        self.loss_fn = HammingLoss()

        print("META:", self.metadata)

    def build_model(self, model_name, arch_params):
        self.model = get_model(
            model_name, out_features=self.metadata["output_features"], in_channels=self.metadata["num_channels"], arch_params=arch_params
        )

    def forward_pass(self, input, true_shortest_paths, train, i):
        output = self.model(input)
        # make grid weights positive
        output = torch.abs(output)
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])

        assert len(weights.shape) == 3, f"{str(weights.shape)}"

        shortest_paths = self.sp_fun(weights, self.lambda_val, self.mode, self.neighbourhood_fn)

        loss = self.loss_fn(shortest_paths, true_shortest_paths)

        logger = self.train_logger if train else self.val_logger

        last_suggestion = {
            "suggested_weights": weights,
            "suggested_path": shortest_paths
        }

        accuracy = (torch.abs(shortest_paths - true_shortest_paths) < 0.5).to(torch.float32).mean()
        extra_loss = self.l1_regconst * torch.mean(output)
        loss += extra_loss

        return loss, accuracy, last_suggestion
