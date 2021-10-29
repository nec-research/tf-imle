#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs if k not in {'d'}])


def to_cmd(c, _path=None):
    command = f'PYTHONPATH=. python3 ./cli/warcraft-cli.py settings/warcraft_shortest_path/12x12_map.json  ' \
              f'trainer_params.mode.type={c["mode_type"]} ' \
              f'trainer_params.mode.objective_type={c["objective_type"]} ' \
              f'trainer_params.mode.use_marginal=true ' \
              f'trainer_params.mode.nb_samples={c["nb_samples"]} ' \
              f'trainer_params.mode.temperature={c["temperature"]} ' \
              f'trainer_params.mode.use_gamma={c["use_gamma"]} ' \
              f'trainer_params.mode.loss_type={c["loss_type"]} ' \
              f'trainer_params.mode.perturb_w={c["perturb_w"]} ' \
              f'trainer_params.mode.perturb_w_prime={c["perturb_w_prime"]} ' \
              f'trainer_params.lambda_val={c["lam"]} ' \
              f'loader_params.data_dir="data/warcraft_shortest_path/{c["k"]}x{c["k"]}" ' \
              f'seed={c["seed"]} evaluate_every=1 use_ray=False'
    return command


def to_logfile(c, path):
    outfile = "{}/temperature_parallel_v8.{}.log".format(path, summary(c).replace("/", "_"))
    return outfile


def main(argv):
    hyp_space = dict(
        mode_type=[1],
        objective_type=['normal'],
        loss_type=['normal'],
        nb_samples=[10],
        lam=[50.0],
        temperature=[0.001],
        use_gamma=[True],
        perturb_w=[True, False],
        perturb_w_prime=[True, False],
        k=[12, 18, 24, 30],
        seed=[0, 1, 2, 3, 4]
    )

    configurations = list(cartesian_product(hyp_space))

    path = 'logs/warcraft/temperature_parallel_v8'
    is_rc = False

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/pminervi/'):
        is_rc = True
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if is_rc is True and os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Epoch: 50' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('{}'.format(command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
