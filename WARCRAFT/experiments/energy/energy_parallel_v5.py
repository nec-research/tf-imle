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
    lhs_str = '--lhs' if c["perturb_lhs"] is True else ''
    rhs_str = '--rhs' if c["perturb_rhs"] is True else ''
    command = f'PYTHONPATH=. python3 ./cli/energy-cost-cli.py ' \
              f'-b {c["batch_size"]} -l {c["learning_rate"]} -e {c["nb_epochs"]} ' \
              f'-m {c["mode"]} -L {c["lbda"]} {lhs_str} {rhs_str} ' \
              f'-t {c["temp"]} -S {c["samples"]} -i {c["gamma_iters"]} ' \
              f'-k {c["gamma_k"]}'
    return command


def to_logfile(c, path):
    outfile = "{}/energy_parallel_v5.{}.log".format(path, summary(c).replace("/", "_"))
    return outfile


def main(argv):
    hyp_space = dict(
        batch_size=[138, 276],
        learning_rate=[0.1, 0.7],
        nb_epochs=[1, 2, 3],
        mode=[0, 1, 2],
        lbda=[1.0, 10.0, 20.0, 50.0, 100.0],
        perturb_lhs=[True, False],
        perturb_rhs=[True, False],
        temp=[0.1, 1.0, 10.0],
        samples=[1, 5, 10],
        gamma_iters=[10],
        gamma_k=[10, 17, 167]
    )

    configurations = list(cartesian_product(hyp_space))

    path = 'logs/energy/energy_parallel_v5'
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
                completed = 'Test mean/std regret' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('{}'.format(command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
