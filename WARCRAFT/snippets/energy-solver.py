#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from maprop.energy.ICON import *
from maprop.energy.get_energy import *
from maprop.energy.intopt_energy_mlp import *

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

print(param)

clf = Gurobi_ICON(relax=False, method=-1, reset=True, presolve=True, **param)
clf.make_model()

print(X_1gtrain.shape, y_train.shape)

_X = X_1gtrain.reshape(-1, 48, 8)
_y = y_train.reshape(-1, 48)
print(_X.shape, _y.shape)

for i in range(_y.shape[0]):
    instance = _X[i, :, :]

    price = _y[i, :]

    # print(instance)
    # print(price)

    print(price)

    res0, _ = clf.solve_model(price)

    print(res0)

    sys.exit(0)

    # print(res0)
    # sys.exit(0)

    res0_scale, _ = clf.solve_model(price * 10.0)

    if np.abs(res0 - res0_scale).sum() > 1e-7:
        print('Scale invariance violated')

    res0_shift, _ = clf.solve_model(price + 10.0)

    if np.abs(res0 - res0_shift).sum() > 1e-7:
        print('Shift invariance violated')
