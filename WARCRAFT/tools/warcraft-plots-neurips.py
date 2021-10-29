#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

name_to_path_template_1 = {
    "I-MLE ($\\mathtt{MAP}-\mathtt{MAP}$)": "logs/warcraft/temperature_beaker_v8/temperature_beaker_v8.k=K_lam=50.0_loss_type=normal_mode_type=1_nb_samples=10_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.001_use_gamma=True.log",
    # "I-MLE ($\\mu-\mathtt{MAP}$)": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=10.0_loss_type=normal_mode_type=1_nb_samples=1_objective_type=normal_perturb_w=True_perturb_w_prime=False_seed=SEED_temperature=0.001_use_gamma=True.log",
    "I-MLE ($\\mu-\\mu$)": "logs/warcraft/temperature_beaker_v11/temperature_beaker_v11.k=K_lam=10.0_loss_type=normal_mode_type=4_nb_samples=1_objective_type=normal_perturb_w=True_perturb_w_prime=True_seed=SEED_temperature=1_use_gamma=True.log",

    "BB ($\lambda = 20$)": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=20.0_loss_type=normal_mode_type=0_nb_samples=10_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.001_use_gamma=True.log",
    # "BB ($\lambda = 10$)": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=10.0_loss_type=normal_mode_type=0_nb_samples=10_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.001_use_gamma=True.log",
    # "BB $\lambda = 1$": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=1.0_loss_type=normal_mode_type=0_nb_samples=10_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.001_use_gamma=True.log",

    ### "BB ($\lambda = 20$, $\\tau = 1$)": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=20.0_loss_type=normal_mode_type=0_nb_samples=1_objective_type=normal_perturb_w=True_perturb_w_prime=True_seed=SEED_temperature=1_use_gamma=True.log",

    "BB ($\lambda = 10$, $\\tau = 1$)": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=10.0_loss_type=normal_mode_type=0_nb_samples=1_objective_type=normal_perturb_w=True_perturb_w_prime=True_seed=SEED_temperature=1_use_gamma=True.log",
    "BB ($\lambda = 100$, $\\tau = 1$)": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=100.0_loss_type=normal_mode_type=0_nb_samples=1_objective_type=normal_perturb_w=True_perturb_w_prime=True_seed=SEED_temperature=1_use_gamma=True.log",
    "BB ($\lambda = 1000$, $\\tau = 1$)": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=1000.0_loss_type=normal_mode_type=0_nb_samples=1_objective_type=normal_perturb_w=True_perturb_w_prime=True_seed=SEED_temperature=1_use_gamma=True.log",
    # "BB ($\lambda = 10000$, $\\tau = 1$)": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=10000.0_loss_type=normal_mode_type=0_nb_samples=1_objective_type=normal_perturb_w=True_perturb_w_prime=True_seed=SEED_temperature=1_use_gamma=True.log",

    "BB ($\lambda = 20$, $\\tau = 0.1$)": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=20.0_loss_type=normal_mode_type=0_nb_samples=1_objective_type=normal_perturb_w=True_perturb_w_prime=True_seed=SEED_temperature=0.1_use_gamma=True.log",
    "BB ($\lambda = 20$, $\\tau = 0.01$)": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=20.0_loss_type=normal_mode_type=0_nb_samples=1_objective_type=normal_perturb_w=True_perturb_w_prime=True_seed=SEED_temperature=0.01_use_gamma=True.log",
    "BB ($\lambda = 20$, $\\tau = 0.001$)": "logs/warcraft/temperature_beaker_v10/temperature_beaker_v10.k=K_lam=20.0_loss_type=normal_mode_type=0_nb_samples=1_objective_type=normal_perturb_w=True_perturb_w_prime=True_seed=SEED_temperature=0.001_use_gamma=True.log",

    # "DPO ($\\sigma=0.1$, $n=1$)": "logs/warcraft/temperature_beaker_v12/temperature_beaker_v12.k=K_lam=0.0_loss_type=normal_mode_type=5_nb_samples=1_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.1_use_gamma=False.log",
    # "DPO ($\\sigma=0.2$, $n=1$)": "logs/warcraft/temperature_beaker_v12/temperature_beaker_v12.k=K_lam=0.0_loss_type=normal_mode_type=5_nb_samples=1_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.2_use_gamma=False.log",
    # "DPO ($\\sigma=0.5$, $n=1$)": "logs/warcraft/temperature_beaker_v12/temperature_beaker_v12.k=K_lam=0.0_loss_type=normal_mode_type=5_nb_samples=1_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.5_use_gamma=False.log",
    # "DPO ($\\sigma=1.0$, $n=1$)": "logs/warcraft/temperature_beaker_v12/temperature_beaker_v12.k=K_lam=0.0_loss_type=normal_mode_type=5_nb_samples=1_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=1.0_use_gamma=False.log",

    # "DPO ($\\sigma=0.1$, $n=10$)": "logs/warcraft/temperature_beaker_v12/temperature_beaker_v12.k=K_lam=0.0_loss_type=normal_mode_type=5_nb_samples=10_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.1_use_gamma=False.log",
    # "DPO ($\\sigma=0.2$, $n=10$)": "logs/warcraft/temperature_beaker_v12/temperature_beaker_v12.k=K_lam=0.0_loss_type=normal_mode_type=5_nb_samples=10_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.2_use_gamma=False.log",
    # "DPO ($\\sigma=0.5$, $n=10$)": "logs/warcraft/temperature_beaker_v12/temperature_beaker_v12.k=K_lam=0.0_loss_type=normal_mode_type=5_nb_samples=10_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.5_use_gamma=False.log",
    # "DPO ($\\sigma=1.0$, $n=10$)": "logs/warcraft/temperature_beaker_v12/temperature_beaker_v12.k=K_lam=0.0_loss_type=normal_mode_type=5_nb_samples=10_objective_type=normal_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=1.0_use_gamma=False.log"

    # "DPO ($\\sigma=0.0$)": "logs/warcraft/temperature_beaker_v14/temperature_beaker_v14.k=K_lam=0.0_loss_type=fy_maximize=False_mode_type=5_nb_samples=1_objective_type=dpo_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.0_use_gamma=False.log",
    ### "DPO ($\\sigma=0.01$)": "logs/warcraft/temperature_beaker_v14/temperature_beaker_v14.k=K_lam=0.0_loss_type=fy_maximize=False_mode_type=5_nb_samples=1_objective_type=dpo_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.01_use_gamma=False.log",
    # "DPO ($\\sigma=0.1$)": "logs/warcraft/temperature_beaker_v14/temperature_beaker_v14.k=K_lam=0.0_loss_type=fy_maximize=False_mode_type=5_nb_samples=1_objective_type=dpo_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.1_use_gamma=False.log",
    # "DPO ($\\sigma=0.5$)": "logs/warcraft/temperature_beaker_v14/temperature_beaker_v14.k=K_lam=0.0_loss_type=fy_maximize=False_mode_type=5_nb_samples=1_objective_type=dpo_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=0.5_use_gamma=False.log",
    # "DPO ($\\sigma=1.0$)": "logs/warcraft/temperature_beaker_v14/temperature_beaker_v14.k=K_lam=0.0_loss_type=fy_maximize=False_mode_type=5_nb_samples=1_objective_type=dpo_perturb_w=False_perturb_w_prime=False_seed=SEED_temperature=1.0_use_gamma=False.log"
}

name_to_path_template_2 = {
    "ResNet18 (Vlastelica et al. 2019)": "logs/warcraft/baseline_beaker_v12/baseline_beaker_v12.k=K_seed=SEED.log"
}

acc_lst = []
seed_lst = []
pt_lst = []
epoch_lst = []
k_lst = []

for name, pt in name_to_path_template_1.items():
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

if False:
    acc_lst += [0.62, 0.88, 0.89, 0.89, 0.9, 0.91, 0.92, 0.9, 0.92, 0.92, 0.92]
    seed_lst += [0] * 11
    pt_lst += ['Perturbed FY (Berthet et al. 2020)'] * 11
    k_lst += [12] * 11
    epoch_lst += [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    acc_lst += [0.25, 0.75, 0.79, 0.80, 0.82, 0.82, 0.835, 0.86, 0.86, 0.86, 0.86]
    seed_lst += [0] * 11
    pt_lst += ['Squared Loss (Berthet et al. 2020)'] * 11
    k_lst += [12] * 11
    epoch_lst += [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

for name, pt in name_to_path_template_2.items():
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

df_dict = {
    'Accuracy': [v * 100 for v in acc_lst],
    'Seed': seed_lst,
    'Method': pt_lst,
    'Epoch': epoch_lst,
    'K': k_lst
}

for k, v in df_dict.items():
    print(k, len(v))

df = pd.DataFrame(df_dict)

for method in name_to_path_template_1.keys():
    for k in [12, 18, 24, 30]:
        if not ('DPO' in method and '$\sigma=0.01$' not in method):
            sub_df = df[df.Epoch == 50]
            sub_df = sub_df[sub_df.Method == method]
            sub_df = sub_df[sub_df.K == k]
            print(f'{method}, k={k}, {sub_df.Accuracy.mean():.1f}, {sub_df.Accuracy.std():.1f}')

# sys.exit(0)

for k in tqdm([12, 12, 18, 24, 30]):
    font_scale = 1.0
    sns.set(font_scale=font_scale)

    sns.set_style("white")
    sns.set_style("ticks")
    plt.grid()

    ax = sns.lineplot(x='Epoch',
                      y='Accuracy',
                      hue='Method',
                      style='Method',
                      data=df[df['K'] == k],
                      linewidth=2.5)

    # ax.legend(handletextpad=0.1)
    plt.setp(ax.get_legend().get_texts(), fontsize='10') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='10') # for legend title

    plt.legend(loc='lower left')

    ax.legend_.set_title(None)

    leg = ax.legend_

    for line in leg.get_lines():
        line.set_linewidth(2.5)

    ax.set_xlim(0, 50)
    ax.set_ylim(0.0, 100.0)

    ax.set_title(f'WarCraft - Accuracy on {k} $\\times$ {k} Maps')

    fig = ax.get_figure()

    # plt.gcf().subplots_adjust(bottom=0.15)
    plt.tight_layout()

    xdim, ydim = 8.0, 3.5
    fig.set_size_inches(xdim, ydim)

    fig.savefig(f'warcraft_neurips_{k}.pdf')

    os.system(f'pdfcrop warcraft_neurips_{k}.pdf')

    plt.clf()
