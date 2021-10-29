#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

clf = SPO_energy(input_size=X_1gtrain.shape[1], param=param, hidden_size=1, optimizer=optim.Adam, lr=0.7,
                 num_layers=1, epochs=4, validation_relax=False, store_validation=True)

clf.fit(X_1gtrain, y_train)
test_rslt = clf.validation_result(X_1gtest,y_test)
intopt_rslt = {'model':'IntOpt','MSE-loss':test_rslt [1],'Regret':test_rslt[0] }
