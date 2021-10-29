#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np

values = []

for line in sys.stdin:
    values += [float(line.rstrip()) * 1]

print(f'mean: {np.mean(values)}')
print(f'std; {np.std(values)}')
