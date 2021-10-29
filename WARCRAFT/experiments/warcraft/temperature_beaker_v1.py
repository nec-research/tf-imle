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
                      f'trainer_params.mode.temperature={c["temperature"]} '
    return command


def to_logfile(c, path):
    outfile = "{}/temperature_beaker_v1.{}.log".format(path, summary(c).replace("/", "_"))
    return outfile


def main(argv):
    hyp_space = dict(
        mode_type=[0, 1, 2, 3],
        objective_type=['normal', 'cost'],
        nb_samples=[1, 10, 100, 1000],
        temperature=[1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    )

    configurations = list(cartesian_product(hyp_space))

    path = 'logs/warcraft/temperature_beaker_v1'
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
                completed = 'Training finished' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)

    header = """#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/array.out
#$ -e $HOME/array.err
#$ -t 1-{}
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true

conda activate gpu

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/workspace/map-backprop

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && sleep 10 && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
