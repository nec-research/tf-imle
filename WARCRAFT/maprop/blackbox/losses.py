# -*- coding: utf-8 -*-

import torch


class HammingLoss(torch.nn.Module):
    def forward(self, suggested, target):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()
        # return (torch.mean(suggested*(1.0-target)) + torch.mean((1.0-suggested)*target)) * 25.0
