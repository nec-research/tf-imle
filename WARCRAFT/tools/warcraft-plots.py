#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

name_to_path_template = {
    "I-MLE ($\\mathtt{MAP}-\mathtt{MAP}$)": "logs/warcraft/temperature_beaker_v8/temperature_beaker_v8.k=K_lam=50.0_loss_type=normal_mode_type=1_nb_samples=10_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.001_use_gamma=True.log",
    "I-MLE ($\\mu-\mathtt{MAP}, \\tau = 10^{-3}$)": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=10.0_loss_type=normal_mode_type=1_nb_samples=1_objective_type=normal_perturb_w=True_perturb_w_prime=False_seed=SEED_temperature=0.001_use_gamma=True.log",
    "I-MLE ($\\mu-\\mu, \\tau = 1$)": "logs/warcraft/temperature_beaker_v11/temperature_beaker_v11.k=K_lam=10.0_loss_type=normal_mode_type=4_nb_samples=1_objective_type=normal_perturb_w=True_perturb_w_prime=True_seed=SEED_temperature=1_use_gamma=True.log",

    # "BB $\lambda = 20$": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=20.0_loss_type=normal_mode_type=0_nb_samples=10_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.001_use_gamma=True.log",
    # "BB $\lambda = 10$": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=10.0_loss_type=normal_mode_type=0_nb_samples=10_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.001_use_gamma=True.log",
    # "BB $\lambda = 1$": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=1.0_loss_type=normal_mode_type=0_nb_samples=10_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.001_use_gamma=True.log",

    "ResNet18": "logs/warcraft/baseline_beaker_v12/baseline_beaker_v12.k=K_seed=SEED.log"
}

acc_lst = []
seed_lst = []
pt_lst = []
epoch_lst = []
k_lst = []

for name, pt in name_to_path_template.items():
    for seed in [0, 1, 2, 3, 4]:
        path = pt.replace('SEED', str(seed))

        for k in [12, 18, 24, 30]:
            k_path = path.replace('K', str(k))

            lines = []
            if os.path.isfile(k_path):
                with open(k_path, 'r') as f:
                    lines = f.readlines()

            epoch_cnt = 0

            for line in lines:
                if 'below_0.0001_percent_acc' in line:
                    line_tokens = line.split()
                    # print(line_tokens)
                    acc_value = float(line_tokens[19].replace(',', ''))

                    acc_lst += [acc_value]
                    seed_lst += [seed]
                    pt_lst += [name]
                    epoch_lst += [epoch_cnt]
                    k_lst += [k]

                    epoch_cnt += 1

df = pd.DataFrame({
    'Accuracy': acc_lst,
    'Seed': seed_lst,
    'Template': pt_lst,
    'Epoch': epoch_lst,
    'K': k_lst
})

for k in tqdm([12, 18, 24, 30]):
    font_scale = 1.6
    sns.set(font_scale=font_scale)

    sns.set_style("white")
    sns.set_style("ticks")
    plt.grid()

    ax = sns.lineplot(x='Epoch',
                      y='Accuracy',
                      hue='Template',
                      style='Template',
                      data=df[df['K'] == k])

    ax.set_xlim(0, 50)
    ax.set_ylim(0.0, 1.0)

    ax.set_title(f'WarCraft - Accuracy on {k} $\\times$ {k} Maps')

    fig = ax.get_figure()

    # plt.gcf().subplots_adjust(bottom=0.15)
    plt.tight_layout()

    xdim, ydim = 8, 4.5
    fig.set_size_inches(xdim, ydim)

    fig.savefig(f'warcraft_{k}.pdf')

    plt.clf()
