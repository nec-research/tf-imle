# -*- coding: utf-8 -*-

from maprop.energy.intopt_energy_mlp import *
from maprop.utils import maybe_parallelize

import pandas as pd
from collections import defaultdict
import numpy as np
import torch

from torch import nn, optim

import functools
from typing import Dict, Any, Callable, List


def sample_sum_of_gammas(nb_samples: int,
                         k: np.ndarray,
                         nb_iterations: int = 10):
    """These should be the epsilons from Th, 1, """
    k_size = k.shape[0]
    samples = np.zeros((nb_samples, k_size), dtype='float')
    for i in range(1, nb_iterations + 1):
        gs = np.random.gamma(1. / k, k / i, size=[nb_samples, k_size])
        samples = samples + gs
    res = ((samples - np.log(nb_iterations)) / k)
    # res = (np.sum([np.random.gamma(1. / k, k / i, size=size) for i in range(1, s + 1)], axis=0) - np.log(s)) / k
    return res


def solve(V: np.ndarray,
          solver: Any) -> np.ndarray:
    assert len(V.shape) == 1 and V.shape[0] == 48
    sol, _ = solver.solve_model(V)
    assert len(sol.shape) == 1 and sol.shape[0] == 48
    return sol


def to_V_lst(n_knap: int,
             n_items: int,
             y_pred_np: np.ndarray) -> List[np.ndarray]:
    V_lst = []
    for kn_nr in range(n_knap):
        kn_start = kn_nr * n_items
        kn_stop = kn_start + n_items
        V = y_pred_np[kn_start:kn_stop]
        V_lst += [V]
    return V_lst


def compute_marginals(V_np: np.ndarray,
                      eps: np.ndarray,
                      solver: Callable) -> List[np.ndarray]:
    batch_size = V_np.shape[0]
    n_items = V_np.shape[1]

    # print('XXX', V_np.shape, eps.shape)

    eps_nb_samples = eps.shape[0]
    eps_batch_size = eps.shape[1]
    eps_n_items = eps.shape[2]

    assert eps_batch_size == batch_size
    assert eps_n_items == n_items

    # EPS = [SAMPLES, BATCH_SIZE, HEIGHT, WIDTH]
    eps_shape = [eps_nb_samples, eps_batch_size, eps_n_items]

    sample_V_np = V_np - eps
    sample_V_np = sample_V_np.reshape([-1, n_items])

    sample_V_lst = [sample_V_np[i] for i in range(sample_V_np.shape[0])]

    sample_sol_lst = map(functools.partial(solve, solver=solver), sample_V_lst)
    sample_sol_lst = list(sample_sol_lst)

    # We solve [SAMPLES, BATCH_SIZE, N_ITEMS] instances of the shortest paths problem
    sample_sol_np = np.asarray(sample_sol_lst)
    sample_sol_np = sample_sol_np.reshape(eps_shape)

    # We compute the mean on the SAMPLE dimension (marginal)
    # [BATCH_SIZE, N_ITEMS]
    marginals = np.mean(sample_sol_np, axis=0)
    # marginals_lst = [marginals[i] for i in range(marginals.shape[0])]

    return marginals


class ICON_MAP(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                y_pred,
                n_items,
                mode: Dict[str, Any],
                lbda,
                solver,
                is_training: bool = True):
        ctx.y_pred = y_pred
        ctx.y_pred_np = ctx.y_pred.detach().cpu().numpy().reshape(-1)
        ctx.n_items = n_items
        ctx.is_training = is_training

        assert mode is not None
        assert lbda is not None

        ctx.mode = mode
        ctx.lbda = lbda

        ctx.solver = solver

        ctx.fwd = None
        ctx.eps = None

        keys = ['use_marginal', 'nb_samples', 'temperature', 'perturb_w', 'perturb_w_prime']
        for key in keys:
            if key not in ctx.mode:
                raise ValueError(f'Missing key {key}')

        n_knap = len(ctx.y_pred_np) // ctx.n_items
        V_lst = to_V_lst(n_knap, ctx.n_items, ctx.y_pred_np)

        if 'use_marginal' in ctx.mode and ctx.mode.use_marginal is True and ctx.is_training is True:
            batch_size = len(V_lst)
            n_items = ctx.n_items

            V_np = np.concatenate([V.reshape(1, -1) for V in V_lst], axis=0)
            assert V_np.shape[1] == n_items

            if 'nb_samples' not in ctx.mode:
                raise ValueError('Number of samples to be used to compute the marginal (mode.nb_samples) unknown')

            nb_samples = ctx.mode.nb_samples

            # EPS = [SAMPLES, BATCH_SIZE, N_ITEMS]
            eps_shape = [nb_samples, batch_size, n_items]

            # Heuristic average path length
            gamma_k = int(167)

            if 'gamma_k' in ctx.mode:
                gamma_k = ctx.mode.gamma_k

            if isinstance(gamma_k, int):
                gamma_k = np.ones(batch_size) * gamma_k

            gamma_iterations = 10
            if 'gamma_iterations' in ctx.mode:
                gamma_iterations = ctx.mode.gamma_iterations

            temperature = ctx.mode.temperature if 'temperature' in ctx.mode else 1.0
            # temperature_prime = ctx.mode.temperature_prime if 'temperature_prime' in ctx.mode else temperature

            # print('AAA', ctx.mode)

            perturb_w = ctx.mode.perturb_w if 'perturb_w' in ctx.mode else True
            perturb_w_prime = ctx.mode.perturb_w_prime if 'perturb_w_prime' in ctx.mode else False

            if perturb_w is True or perturb_w_prime is True:
                _nb_gamma_samples = nb_samples * n_items
                eps = sample_sum_of_gammas(_nb_gamma_samples, gamma_k, gamma_iterations)
                ctx.eps = eps.reshape(eps_shape)

            if perturb_w is True:
                ctx.lhs = compute_marginals(V_np, ctx.eps * temperature, ctx.solver)
                ctx.fwd = ctx.lhs

        if ctx.fwd is None:
            sol_lst = map(functools.partial(solve, solver=ctx.solver), V_lst)
            sol_lst = list(sol_lst)
            ctx.fwd = np.concatenate([x.reshape(1, -1) for x in sol_lst], axis=0)

        return torch.from_numpy(ctx.fwd).float().to(ctx.y_pred.device)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.shape == ctx.fwd.shape
        grad_output_numpy = grad_output.detach().cpu().numpy()

        if ctx.mode.type == 0:
            y_prime_np = ctx.y_pred_np.reshape(-1) + ctx.lbda * grad_output_numpy.reshape(-1)
        elif ctx.mode.type == 1:
            y_prime_np = ctx.lbda * grad_output_numpy.reshape(-1)
        elif ctx.mode.type == 2:
            y_prime_np = grad_output_numpy.reshape(-1)
        else:
            assert False, f'Unknown model {ctx.mode}'

        n_knap = len(y_prime_np) // ctx.n_items
        V_prime_lst = to_V_lst(n_knap, ctx.n_items, y_prime_np)
        V_prime_np = np.concatenate([V.reshape(1, -1) for V in V_prime_lst], axis=0)

        lhs = ctx.fwd
        rhs = None

        if 'use_marginal' in ctx.mode and ctx.mode.use_marginal is True:
            temperature = ctx.mode.temperature if 'temperature' in ctx.mode else 1.0
            temperature_prime = ctx.mode.temperature_prime if 'temperature_prime' in ctx.mode else temperature

            # Solutions identified during forward inference, MAP[\theta]
            lhs = ctx.fwd

            # Solutions identified with the perturbed weights, MAP[\theta']
            # rhs = fwd_prime

            perturb_w = ctx.mode.perturb_w if 'perturb_w' in ctx.mode else True
            perturb_w_prime = ctx.mode.perturb_w_prime if 'perturb_w_prime' in ctx.mode else False

            if perturb_w is True:
                lhs = ctx.lhs

            if perturb_w_prime is True:
                rhs = compute_marginals(V_prime_np, ctx.eps * temperature_prime, ctx.solver)

        if rhs is None:
            sol_prime_lst = map(functools.partial(solve, solver=ctx.solver), V_prime_lst)
            sol_prime_lst = list(sol_prime_lst)
            rhs = np.concatenate([x.reshape(1, -1) for x in sol_prime_lst], axis=0)

        gradient = - (lhs - rhs) / ctx.lbda
        gradient = gradient.reshape(-1, 1)

        return torch.from_numpy(gradient).to(grad_output.device), None, None, None, None, None


class maprop_energy:
    def __init__(self, param,
                 input_size, hidden_size, num_layers, target_size=1,
                 doScale=True, n_items=48, epochs=1, batchsize=24,
                 verbose=False, validation_relax=True,
                 optimizer=optim.Adam, model_save=False, model_name=None,
                 model=None, store_validation=False, scheduler=False,
                 mode=None,
                 lbda=None,
                 weight_decay=0.0,
                 evaluate=None,
                 **hyperparams):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.param = param
        self.doScale = doScale
        self.n_items = n_items
        self.epochs = epochs
        self.batchsize = batchsize

        self.verbose = verbose
        self.validation_relax = validation_relax
        # self.test_relax = test_relax
        self.optimizer = optimizer
        self.model_save = model_save
        self.model_name = model_name
        self.hyperparams = hyperparams
        self.store_validation = store_validation
        self.scheduler = scheduler

        self.mode = mode
        self.lbda = lbda

        self.weight_decay = weight_decay
        self.evaluate = evaluate

        self.model = MultilayerRegression(input_size=input_size, hidden_size=hidden_size, target_size=target_size, num_layers=num_layers)

        print('Model state:')
        for param_tensor in self.model.state_dict():
            print(f'\t{param_tensor}\t{self.model.state_dict()[param_tensor].size()}')

        self.optimizer = optimizer(self.model.parameters(), weight_decay=self.weight_decay, **hyperparams)

        self.maprop_clf = Gurobi_ICON(relax=False, method=-1, reset=True, presolve=True, **self.param)
        self.maprop_clf.make_model()

    def fit(self, X, y, X_validation=None, y_validation=None, X_test=None, y_test=None):
        self.model_time = 0.
        runtime = 0.

        validation_time = 0
        test_time = 0
        # if validation true validation and tets data should be provided

        validation = (X_validation is not None) and (y_validation is not None)
        test = (X_test is not None) and (y_test is not None)
        param = self.param

        if self.doScale:
            self.scaler = preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X)

        if validation:
            start_validation = time.time()

            if self.doScale:
                X_validation = self.scaler.transform(X_validation)
            end_validation = time.time()
            validation_time += end_validation - start_validation

        if test:
            start_test = time.time()

            if self.doScale:
                X_test = self.scaler.transform(X_test)
            end_test = time.time()
            test_time += end_test - start_test

        validation_relax = self.validation_relax

        criterion = nn.MSELoss(reduction='mean')
        n_items = self.n_items
        epochs = self.epochs

        batchsize = self.batchsize
        n_batches = X.shape[0] // (batchsize * n_items)

        print('N_BATCHES', n_batches)

        n_knapsacks = X.shape[0] // n_items

        subepoch = 0
        validation_result = []
        shuffled_batches = [i for i in range(n_batches)]

        n_train = len(y)
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: 1 if x < 2 else 0.95 ** x)

        if self.evaluate is not None:
            print(f'BEFORE TRAINING')
            self.evaluate()

        for e in range(epochs):
            logging.info('Epoch %d' % e)
            np.random.shuffle(shuffled_batches)
            for i in range(n_batches):
                start = time.time()

                n_start = (batchsize * shuffled_batches[i] * n_items)
                n_stop = (batchsize * (shuffled_batches[i] + 1) * n_items)

                # [1152, 8], 1152 = 24 * 48 = batch_size * n_items
                X_tensor = torch.from_numpy(X[n_start:n_stop, :]).float()

                # [1152, 1]
                y_target = torch.from_numpy(y[n_start:n_stop][:, np.newaxis]).float()

                self.optimizer.zero_grad()

                # [1152, 1]
                y_pred = self.model(X_tensor)

                ICON_MAP_fun = ICON_MAP.apply

                maprop_clf = self.maprop_clf
                
                is_training = self.model.training

                # [24, 48]
                sol_pred = ICON_MAP_fun(y_pred, self.n_items, self.mode, self.lbda, maprop_clf, is_training)

                # [24]
                dot = (y_target.view(-1) * sol_pred.view(-1)).reshape(self.batchsize, -1).sum(axis=1)

                loss = torch.mean(dot)

                loss.backward()
                self.optimizer.step()

                end = time.time()
                runtime += end - start
                subepoch += 1

                print(f'Epoch[{e + 1}/{i + 1}], loss(train):{loss.item():.2f} @ {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ')

                if ((i + 1) % 7 == 0) | ((i + 1) % n_batches == 0):
                    if self.model_save:
                        torch.save(self.model.state_dict(), str(self.model_name + "_Epoch" + str(e) + "_" + str(i) + ".pth"))

                    if self.store_validation:
                        if validation:
                            y_pred_validation = self.predict(X_validation, doScale=False)
                            if not hasattr(self, 'sol_validation'):
                                # print('EVAL', 'y_validation', y_validation.shape)
                                self.sol_validation = ICON_solution(param=param, y=y_validation,
                                                                    relax=self.validation_relax, n_items=self.n_items)
                        else:
                            self.sol_validation = None
                            y_pred_validation = None

                        if test:
                            y_pred_test = self.predict(X_test, doScale=False)
                            if not hasattr(self, 'sol_test'):
                                self.sol_test = ICON_solution(param=param, y=y_test, relax=False, n_items=self.n_items)
                        else:
                            self.sol_test = None
                            y_pred_test = None

                        dict_validation = validation_module(param=param, n_items=self.n_items,
                                                            run_time=runtime, epoch=e, batch=i,
                                                            model_time=self.model_time,
                                                            y_target_validation=y_validation,
                                                            sol_target_validation=self.sol_validation,
                                                            y_pred_validation=y_pred_validation,
                                                            y_target_test=y_test, sol_target_test=self.sol_test,
                                                            y_pred_test=y_pred_test,
                                                            relax=self.validation_relax)

                        print(dict_validation)

                        validation_result.append(dict_validation)

            if self.evaluate is not None:
                print(f'EPOCH {e}')
                self.evaluate()

        if self.store_validation:
            # return test_result
            dd = defaultdict(list)
            for d in validation_result:
                for key, value in d.items():
                    dd[key].append(value)
            df = pd.DataFrame.from_dict(dd)
            # self.logger.info('Completion Time %s \n' %str(datetime.datetime.now()) )
            logging.info('Completion Time %s \n' % str(datetime.datetime.now()))

            return df

    def validation_result(self, X_validation, y_validation, scaler=None, doScale=True):
        if doScale:
            if scaler is None:
                try:
                    scaler = self.scaler
                except:
                    raise Exception("you asked to do scaler but no StandardScaler found")
            X_validation = scaler.transform(X_validation)

        model = self.model
        model.eval()
        X_tensor = torch.tensor(X_validation, dtype=torch.float)
        y_pred = model(X_tensor).detach().numpy().squeeze()
        model.train()
        sol_validation = ICON_solution(param=self.param, y=y_validation, relax=False, n_items=self.n_items)

        validation_rslt = validation_module(param=self.param, n_items=self.n_items,
                                            y_target_test=y_validation, sol_target_test=sol_validation,
                                            y_pred_test=y_pred)

        return validation_rslt['test_regret'], validation_rslt['test_mse']

    def predict(self, X, scaler=None, doScale=True):
        if doScale:
            if scaler is None:
                try:
                    scaler = self.scaler
                except:
                    raise Exception("you asked to do scaler but no StandardScaler found")
            X1 = scaler.transform(X)
        else:
            X1 = X

        model = self.model
        model.eval()
        X_tensor = torch.tensor(X1, dtype=torch.float)
        y_pred = model(X_tensor).detach().numpy().squeeze()
        model.train()
        return y_pred
