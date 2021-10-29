#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from maprop.energy.maprop import *

from maprop.energy.get_energy import *
from maprop.energy.ICON import *

import numpy as np
import multiprocessing
import torch

import argparse

import logging

formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='ICONexp.log', level=logging.INFO,format=formatter)

cpu_count = multiprocessing.cpu_count()
# np.set_num_threads(cpu_count)
torch.set_num_threads(cpu_count)


class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self


def _run_experiment(X_1gtrain, y_train, X_1gvalidation, y_validation, X_1gtest, y_test,
					param, batch_size, learning_rate, nb_epochs, mode, lbda,
					weight_decay, num_layers, is_debug):
	hidden_size = 1

	if num_layers > 1:
		hidden_size = 100

	def _evaluate():
		val_rslt = clf.validation_result(X_1gvalidation, y_validation)
		val_maprop_rslt = {'model': 'MAProp', 'MSE-loss': val_rslt[1], 'Regret': val_rslt[0]}

		print('Dev', val_maprop_rslt)

		test_rslt = clf.validation_result(X_1gtest, y_test)
		maprop_rslt = {'model': 'MAProp', 'MSE-loss': test_rslt[1], 'Regret': test_rslt[0]}

		print('Test', maprop_rslt)
		return val_maprop_rslt, maprop_rslt

	clf = maprop_energy(input_size=X_1gtrain.shape[1], param=param, hidden_size=hidden_size, batchsize=batch_size, optimizer=optim.Adam,
						lr=learning_rate, num_layers=num_layers, epochs=nb_epochs, validation_relax=False, store_validation=False,
						mode=mode, lbda=lbda, weight_decay=weight_decay, evaluate=_evaluate if is_debug is True else None)

	print('Train X y shapes', X_1gtrain.shape, y_train.shape)

	clf.fit(X_1gtrain, y_train, X_validation=X_1gvalidation, y_validation=y_validation, X_test=X_1gtest, y_test=y_test)

	val_maprop_rslt, maprop_rslt = _evaluate()

	return val_maprop_rslt['Regret'], maprop_rslt['Regret']


def run_experiment(X_1gtrain, y_train, X_1gvalidation, y_validation, X_1gtest, y_test,
				   param, batch_size, learning_rate, nb_epochs, mode, lbda, nb_loops,
				   weight_decay, num_layers, is_debug):
	val_regret_values = []
	test_regret_values = []
	for i in range(nb_loops):
		val_regret, test_regret = _run_experiment(X_1gtrain, y_train, X_1gvalidation, y_validation, X_1gtest, y_test,
												  param, batch_size, learning_rate, nb_epochs, mode, lbda, weight_decay,
												  num_layers, is_debug)
		val_regret_values += [val_regret]
		test_regret_values += [test_regret]
	return val_regret_values, test_regret_values


def main(argv):
	parser = argparse.ArgumentParser('Energy MAProp', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--batch-size', '-b', action='store', type=int, default=24)
	parser.add_argument('--learning-rate', '-l', action='store', type=float, default=0.001)
	parser.add_argument('--epochs', '-e', action='store', type=int, default=10)

	parser.add_argument('--mode-type', '-m', action='store', type=int, default=0)
	parser.add_argument('--gamma-k', '-k', action='store', type=int, default=167)
	parser.add_argument('--gamma-iterations', '-i', action='store', type=int, default=10)
	parser.add_argument('--temperature', '-t', action='store', type=float, default=10.0)

	parser.add_argument('--perturb-lhs', '--lhs', action='store_true', default=False)
	parser.add_argument('--perturb-rhs', '--rhs', action='store_true', default=False)
	parser.add_argument('--nb-samples', '-S', action='store', type=int, default=10)

	parser.add_argument('--lbda', '-L', action='store', type=float, default=100.0)
	parser.add_argument('--weight-decay', '-w', action='store', type=float, default=0.0)
	parser.add_argument('--num-layers', '-n', action='store', type=int, default=1)

	parser.add_argument('--loops', action='store', type=int, default=10)

	parser.add_argument('--debug', '-D', action='store_true', default=False)

	args = parser.parse_args(argv)

	batch_size = args.batch_size
	learning_rate = args.learning_rate
	nb_epochs = args.epochs

	mode_type = args.mode_type
	gamma_k = args.gamma_k
	gamma_iterations = args.gamma_iterations
	temperature = args.temperature

	perturb_lhs = args.perturb_lhs
	perturb_rhs = args.perturb_rhs
	nb_samples = args.nb_samples

	lbda = args.lbda
	weight_decay = args.weight_decay
	num_layers = args.num_layers

	nb_loops = args.loops

	is_debug = args.debug

	mode_dict = AttrDict({
		'type': mode_type,
		'use_marginal': True,
		'gamma_k': gamma_k,
		'gamma_iterations': gamma_iterations,
		'temperature': temperature,
		'perturb_w': perturb_lhs,
		'perturb_w_prime': perturb_rhs,
		'nb_samples': nb_samples
	})

	(X_1gtrain, y_train, X_1gtest, y_test) = get_energy("data/intopt/energy/prices2013.dat")
	X_1gvalidation = X_1gtest[0:2880, :]
	y_validation = y_test[0:2880]
	y_test = y_test[2880:]
	X_1gtest = X_1gtest[2880:, :]
	weights = [[1 for i in range(48)]]
	weights = np.array(weights)
	X_1gtrain = X_1gtrain[:, 1:]
	X_1gvalidation = X_1gvalidation[:, 1:]
	X_1gtest = X_1gtest[:, 1:]

	file = "data/intopt/energy/load1/day01.txt"
	param = data_reading(file)

	val_regrets, test_regrets = run_experiment(X_1gtrain, y_train, X_1gvalidation, y_validation, X_1gtest, y_test,
											   param, batch_size, learning_rate, nb_epochs, mode_dict, lbda, nb_loops,
											   weight_decay, num_layers, is_debug)

	print('All validation regrets:', val_regrets)
	print('All test regrets:', test_regrets)

	print('Validation mean/std regret:', np.mean(val_regrets), np.std(val_regrets))
	print('Test mean/std regret:', np.mean(test_regrets), np.std(test_regrets))

	return


if __name__ == '__main__':
	main(sys.argv[1:])
