# -*- coding: utf-8 -*-

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, probs_to_logits, lazy_property, logits_to_probs

class SumOfGamma(Distribution):
    arg_constraints = {'total_count': constraints.nonnegative_integer,
                       'probs': constraints.unit_interval,
                       'logits': constraints.real}
    has_enumerate_support = True
