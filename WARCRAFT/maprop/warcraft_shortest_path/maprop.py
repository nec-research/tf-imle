# -*- coding: utf-8 -*-

import numpy as np
import torch

from torch import Tensor

from maprop.blackbox.losses import HammingLoss

from maprop.warcraft_shortest_path.trainers import ShortestPathAbstractTrainer

from maprop.blackbox.dijkstra import get_solver
from maprop.utils import maybe_parallelize

from maprop.models import get_model

from maprop import perturbations
from maprop import fenchel_young as fy

from typing import Dict, Any, Callable


def sample_sum_of_gammas(nb_samples: int,
                         k: np.ndarray,
                         nb_iterations: int = 10):
    """These should be the epsilons from Th, 1, """
    k_size = k.shape[0]
    samples = np.zeros((nb_samples, k_size), dtype='float')
    for i in range(1, nb_iterations + 1):
        gs = np.random.gamma(1. / k, k / i, size=[nb_samples, k_size])
        samples = samples + gs
    # res = temperature * ((samples - np.log(nb_iterations)) / k)
    res = ((samples - np.log(nb_iterations)) / k)
    # res = (np.sum([np.random.gamma(1. / k, k / i, size=size) for i in range(1, s + 1)], axis=0) - np.log(s)) / k
    return res


def translate_weights(weights: np.ndarray) -> np.ndarray:
    # Weights can be negative - shift them so they are positive
    batch_size = weights.shape[0]
    res = (weights.T - np.minimum(np.amin(weights.reshape(batch_size, -1), axis=-1), 0).T).T
    return res


def compute_marginals(weights: np.ndarray,
                      eps: np.ndarray,
                      solver: Callable):
    batch_size = weights.shape[0]
    height_size = weights.shape[1]
    width_size = weights.shape[2]

    eps_nb_samples = eps.shape[0]
    eps_batch_size = eps.shape[1]
    eps_height_size = eps.shape[2]
    eps_width_size = eps.shape[3]

    assert eps_batch_size == batch_size
    assert eps_height_size == height_size
    assert eps_width_size == width_size

    # EPS = [SAMPLES, BATCH_SIZE, HEIGHT, WIDTH]
    eps_shape = [eps_nb_samples, batch_size, height_size, width_size]

    # [BATCH_SIZE, HEIGHT, WIDTH] - [SAMPLES, BATCH_SIZE, HEIGHT, WIDTH]
    # will have shape [SAMPLES, BATCH_SIZE, HEIGHT, WIDTH] thanks to broadcasting.
    # Note that the parameters of the Exponential Family distribution are - theta, so we have to take
    # the negative of the Gumbel sample to perturb-and-MAP.

    # sample_weights = orig_weights - eps
    sample_weights = weights - eps
    sample_weights = sample_weights.reshape([-1, height_size, width_size])

    # We solve [SAMPLES, BATCH_SIZE, HEIGHT, WIDTH] instances of the shortest paths problem
    sample_tours = np.asarray(maybe_parallelize(solver, arg_list=list(sample_weights)))
    sample_tours = sample_tours.reshape(eps_shape)

    # We compute the mean on the SAMPLE dimension (marginal)
    marginals = np.mean(sample_tours, axis=0)
    return marginals


class ShortestPathMAP(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                weights: Tensor,
                lambda_val: float,
                mode: Dict[str, Any],
                neighbourhood_fn: str = "8-grid",
                is_training: bool = True,
                true_weights: np.ndarray = None):
        ctx.lambda_val = lambda_val
        ctx.neighbourhood_fn = neighbourhood_fn
        ctx.solver = get_solver(neighbourhood_fn)

        ctx.mode = mode
        ctx.is_training = is_training

        ctx.true_weights = true_weights
        if ctx.true_weights is not None:
            ctx.true_weights = ctx.true_weights.detach().cpu().numpy()

        # [BATCH, HEIGHT, WIDTH]
        ctx.weights = weights.detach().cpu().numpy()
        ctx.eps = None

        # What will be returned by the function
        ctx.suggested_tours = None

        keys = ['use_marginal', 'nb_samples', 'use_gamma', 'temperature', 'perturb_w', 'perturb_w_prime']
        for key in keys:
            if key not in ctx.mode:
                raise ValueError(f'Missing key {key}')

        # Sample the eps term for computing the marginals
        if 'use_marginal' in ctx.mode and ctx.mode.use_marginal is True and ctx.is_training is True:
            batch_size = ctx.weights.shape[0]
            height_size = ctx.weights.shape[1]
            width_size = ctx.weights.shape[2]

            if 'nb_samples' not in ctx.mode:
                raise ValueError('Number of samples to be used to compute the marginal (mode.nb_samples) unknown')

            nb_samples = ctx.mode.nb_samples

            # EPS = [SAMPLES, BATCH_SIZE, HEIGHT, WIDTH]
            eps_shape = [nb_samples, batch_size, height_size, width_size]

            perturb_w = ctx.mode.perturb_w if 'perturb_w' in ctx.mode else True
            perturb_w_prime = ctx.mode.perturb_w_prime if 'perturb_w_prime' in ctx.mode else False

            if 'use_gamma' in ctx.mode and ctx.mode.use_gamma is True:
                # Heuristic average path length
                gamma_k = int(height_size * 1.3)

                if 'gamma_per_instance' in ctx.mode and ctx.mode.gamma_per_instance is True:
                    # Another strategy: batch-wise k estimate, per-sample k (this one)
                    gamma_k = ctx.suggested_tours.reshape(batch_size, -1).sum(axis=1)

                if 'gamma_k' in ctx.mode:
                    gamma_k = ctx.mode.gamma_k

                gamma_iterations = 10
                if 'gamma_iterations' in ctx.mode:
                    gamma_iterations = ctx.mode.gamma_iterations

                if isinstance(gamma_k, int):
                    gamma_k = np.ones(batch_size) * gamma_k

                # [SAMPLES, BATCH_SIZE, HEIGHT, WIDTH]
                if perturb_w is True or perturb_w_prime is True:
                    _nb_gamma_samples = nb_samples * height_size * width_size
                    eps = sample_sum_of_gammas(_nb_gamma_samples, gamma_k, gamma_iterations)
                    ctx.eps = eps.reshape(eps_shape)
            else:
                # [SAMPLES, BATCH_SIZE, HEIGHT, WIDTH]
                ctx.eps = np.random.gumbel(loc=0.0, scale=1.0, size=eps_shape)

            temperature = ctx.mode.temperature if 'temperature' in ctx.mode else 1.0
            # temperature_prime = ctx.mode.temperature_prime if 'temperature_prime' in ctx.mode else temperature

            if perturb_w is True:
                ctx.lhs = compute_marginals(ctx.weights, ctx.eps * temperature, ctx.solver)
                ctx.suggested_tours = ctx.lhs

        if ctx.suggested_tours is None:
            ctx.suggested_tours = np.asarray(maybe_parallelize(ctx.solver, arg_list=list(ctx.weights)))

        return torch.from_numpy(ctx.suggested_tours).float().to(weights.device)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.shape == ctx.suggested_tours.shape
        grad_output_numpy = grad_output.detach().cpu().numpy()

        # We should not need to compute any backward pass during evaluation
        assert ctx.is_training is True

        # [BATCH, HEIGHT, WIDTH]
        if ctx.mode.type == 0:
            weights_prime = ctx.weights + ctx.lambda_val * grad_output_numpy
        elif ctx.mode.type == 1:
            weights_prime = ctx.lambda_val * grad_output_numpy
        elif ctx.mode.type == 2:
            weights_prime = grad_output_numpy
        elif ctx.mode.type == 4:
            assert ctx.true_weights is not None
            weights_prime = ctx.lambda_val * grad_output_numpy
        elif ctx.mode.type == 6:
            assert ctx.true_weights is not None
            weights_prime = ctx.weights + ctx.lambda_val * grad_output_numpy
        else:
            raise ValueError(f'Unknown mode type {ctx.mode}')

        # Make sure weights are positive by translating them
        weights_prime = translate_weights(weights_prime)

        # Weights can be negative - shift them so they are positive
        # XXX: this is already done by translate_weights no?
        batch_size = weights_prime.shape[0]
        weights_prime = (weights_prime.T - np.minimum(np.amin(weights_prime.reshape(batch_size, -1), axis=-1), 0).T).T

        lhs = ctx.suggested_tours
        rhs = None

        # better_paths = np.asarray(maybe_parallelize(ctx.solver, arg_list=list(weights_prime)))
        # gradient = - (ctx.suggested_tours - better_paths) / ctx.lambda_val

        if 'use_marginal' in ctx.mode and ctx.mode.use_marginal is True:
            temperature = ctx.mode.temperature if 'temperature' in ctx.mode else 1.0
            temperature_prime = ctx.mode.temperature_prime if 'temperature_prime' in ctx.mode else temperature

            # Paths identified during forward inference, MAP[\theta]
            lhs = ctx.suggested_tours

            # Paths identified with the perturbed weights, MAP[\theta']
            # rhs = better_paths

            perturb_w = ctx.mode.perturb_w if 'perturb_w' in ctx.mode else True
            perturb_w_prime = ctx.mode.perturb_w_prime if 'perturb_w_prime' in ctx.mode else False

            if perturb_w is True:
                lhs = ctx.lhs
                # lhs = compute_marginals(orig_weights, ctx.eps * temperature, ctx.solver)

            if perturb_w_prime is True:
                if ctx.mode.type in {4}:
                    rhs = compute_marginals(ctx.true_weights, ctx.eps * temperature_prime, ctx.solver)
                else:
                    rhs = compute_marginals(weights_prime, ctx.eps * temperature_prime, ctx.solver)
                    # rhs_true = compute_marginals(ctx.true_weights, ctx.eps * temperature_prime, ctx.solver)

                    # print('GRAD', grad_output_numpy[0])
                    # print('WEIGHTS_PRIME', weights_prime[0])
                    # print('TRUE WEIGHTS', ctx.true_weights[0])

                    # print('RHS', rhs[0])
                    # print('RHS TRUE', rhs_true[0])

                    # print('LHS', lhs[0])

                    # import sys
                    # sys.exit(0)

            # gradient = - (marginals - better_paths) / ctx.lambda_val
            # gradient = - (lhs - rhs) / ctx.lambda_val

        if rhs is None:
            rhs = np.asarray(maybe_parallelize(ctx.solver, arg_list=list(weights_prime)))

        gradient = - (lhs - rhs) / ctx.lambda_val

        return torch.from_numpy(gradient).to(grad_output.device), None, None, None, None, None


class DijkstraMAP(ShortestPathAbstractTrainer):
    def __init__(self, *, l1_regconst, lambda_val,
                 mode,
                 **kwargs):
        super().__init__(**kwargs)
        self.l1_regconst = l1_regconst
        self.lambda_val = lambda_val

        self.mode = mode
        print(f'MAP-BACKPROP MODE: {self.mode}')

        self.sp_fun = ShortestPathMAP.apply

        print('loss_type' in self.mode)

        def bb_dijkstra(_weights: Tensor) -> Tensor:
            _solver = get_solver(self.neighbourhood_fn)
            _weights_np = _weights.detach().cpu().numpy()
            _paths = np.asarray(maybe_parallelize(_solver, arg_list=list(_weights_np)))
            _res = torch.from_numpy(_paths).float().to(_weights.device)
            return _res

        self.bb_dijkstra = bb_dijkstra

        if 'loss_type' in self.mode and self.mode.loss_type in {'mse'}:
            print(f'LOSS FUNCTION: {self.mode.loss_type}')
            self.loss_fn = torch.nn.MSELoss(reduction='sum')
        elif 'loss_type' in self.mode and self.mode.loss_type in {'fy'}:
            is_maximize = True

            if 'dpo' in self.mode:
                if 'maximize' in self.mode.dpo:
                    if isinstance(self.mode.dpo.maximize, bool):
                        is_maximize = self.mode.dpo.maximize
                    elif isinstance(self.mode.dpo.maximize, str):
                        is_maximize = self.mode.dpo.maximize in {'true', 'True'}
                    else:
                        assert False, f'self.mode.dpo.maximize has the wrong type: {type(self.mode.dpo.maximize)}'

            loss_fn = fy.FenchelYoungLoss(self.bb_dijkstra,
                                          num_samples=self.mode.nb_samples,
                                          sigma=self.mode.temperature,
                                          batched=True,
                                          maximize=is_maximize)

            self.loss_fn = lambda x, y: torch.mean(loss_fn(x, y))
        else:
            self.loss_fn = HammingLoss()

        if 'objective_type' in self.mode and self.mode.objective_type in {'cost', 'cost2'}:
            print(f'OBJECTIVE TYPE: {self.mode.objective_type}')

        print("META:", self.metadata)

    def build_model(self, model_name, arch_params):
        self.model = get_model(
            model_name, out_features=self.metadata["output_features"], in_channels=self.metadata["num_channels"], arch_params=arch_params
        )

    def forward_pass(self, input, true_shortest_paths, train, i, true_weights=None):
        output = self.model(input)
        # make grid weights positive
        output = torch.abs(output)
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])

        assert len(weights.shape) == 3, f"{str(weights.shape)}"

        # if true_weights is not None:
        #     print(type(true_weights), true_weights.shape)

        is_training = self.model.training

        use_fenchel_young = False
        if 'loss_type' in self.mode and self.mode.objective_type in {'fy'}:
            use_fenchel_young = True

        p_dijkstra = None

        if self.mode.type == 5:
            p_dijkstra = self.bb_dijkstra

            # If we use Fenchel-Young, the perturbation is being done by the loss function
            if is_training is True and use_fenchel_young is False:
                p_dijkstra = perturbations.perturbed(self.bb_dijkstra,
                                                     num_samples=self.mode.nb_samples,
                                                     sigma=self.mode.temperature,
                                                     noise='gumbel',
                                                     batched=True,
                                                     device=weights.device)

            shortest_paths = p_dijkstra(weights)

            # print('XXX', is_training, weights.shape, shortest_paths.shape, true_shortest_paths.shape)

        else:
            shortest_paths = self.sp_fun(weights, self.lambda_val, self.mode,
                                         self.neighbourhood_fn, is_training, true_weights)

        if 'objective_type' in self.mode and self.mode.objective_type in {'cost'}:
            #  Luca's first MSE objective - weird because uses true_weights, try without
            path_costs_mat = true_weights * shortest_paths
            path_costs = torch.sum(path_costs_mat, dim=(1, 2))

            true_costs_mat = true_weights * true_shortest_paths
            true_path_costs = torch.sum(true_costs_mat, dim=(1, 2))

            loss = self.loss_fn(path_costs, true_path_costs)
        elif 'objective_type' in self.mode and self.mode.objective_type in {'cost2'}:
            #  This objective uses inferred weights to compute the inferred cost
            path_costs_mat = weights * shortest_paths
            path_costs = torch.sum(path_costs_mat, dim=(1, 2))

            true_costs_mat = true_weights * true_shortest_paths
            true_path_costs = torch.sum(true_costs_mat, dim=(1, 2))

            loss = self.loss_fn(path_costs, true_path_costs)
        elif 'objective_type' in self.mode and self.mode.objective_type in {'berthet', 'dpo'}:
            loss = self.loss_fn(weights, true_shortest_paths)
        else:
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


# XXX: Subsumed by DijkstraMAP, remove?
class DijkstraMAPSquaredError(ShortestPathAbstractTrainer):
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

        self.loss_fn = torch.nn.MSELoss()

        print("META:", self.metadata)

    def build_model(self, model_name, arch_params):
        self.model = get_model(
            model_name, out_features=self.metadata["output_features"], in_channels=self.metadata["num_channels"], arch_params=arch_params
        )

    def forward_pass(self, input, true_shortest_paths, train, i, true_weights=None):
        output = self.model(input)
        # make grid weights positive
        output = torch.abs(output)
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])

        assert len(weights.shape) == 3, f"{str(weights.shape)}"

        shortest_paths = self.sp_fun(weights, self.lambda_val, self.mode, self.neighbourhood_fn)

        #  added here
        path_costs_mat = true_weights*shortest_paths
        path_costs = torch.sum(path_costs_mat, dim=(1, 2))

        true_costs_mat = true_weights*true_shortest_paths
        true_path_costs = torch.sum(true_costs_mat, dim=(1, 2))

        # prints to delete!!

        # print('shortest', shortest_paths.shape, shortest_paths, sep='\n')
        # print('costs', true_weights, sep='\n' )
        # print('path times cost', path_costs, sep='\n')
        # print('sum of path cost?', path_costs,  sep='\n')
        # print('sum true path costs', true_path_costs, sep='\n')
        # print('-'*50)
        # # ok this looks the right one!

        loss = self.loss_fn(path_costs, true_path_costs)

        logger = self.train_logger if train else self.val_logger

        last_suggestion = {
            "suggested_weights": weights,
            "suggested_path": shortest_paths
        }

        accuracy = (torch.abs(shortest_paths - true_shortest_paths) < 0.5).to(torch.float32).mean()
        extra_loss = self.l1_regconst * torch.mean(output)
        loss += extra_loss

        return loss, accuracy, last_suggestion
