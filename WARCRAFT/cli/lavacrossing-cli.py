#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from logging import WARNING
import warnings

import psutil
import ray
import re

import torch

import getpass

from maprop.logger import Logger
from maprop.utils import set_seed, save_metrics_params, update_params_from_cmdline, save_settings_to_json

import maprop.lavacrossing.data_utils as lavacrossing_data

from maprop.utils import load_json

from maprop.lavacrossing.trainers import BaselineTrainer, DijkstraOnFull
from maprop.lavacrossing.maprop import DijkstraMAP


warnings.filterwarnings("ignore")


def get_trainer(trainer_name):
    trainers = {
        "Baseline": BaselineTrainer,
        "DijkstraOnFull": DijkstraOnFull,
        "DijkstraMAP": DijkstraMAP
    }
    return trainers[trainer_name]


dataset_loaders = {
    "lavacrossing": lavacrossing_data.load_dataset
}

trainer_loaders = {
    "lavacrossing": get_trainer
}

required_top_level_params = [
    "model_dir",
    "seed",
    "loader_params",
    "problem_type",
    "trainer_name",
    "trainer_params",
    "num_epochs",
    "evaluate_every",
    "save_visualizations"
]

optional_top_level_params = ["num_cpus", "use_ray", "default_json", "id", "fast_mode", "fast_forward_training"]


def verify_top_level_params(**kwargs):
    for kwarg in kwargs:
        if kwarg not in required_top_level_params and kwarg not in optional_top_level_params:
            raise ValueError("Unknown top_level argument: {}".format(kwarg))

    for required in required_top_level_params:
        if required not in kwargs.keys():
            raise ValueError("Missing required argument: {}".format(required))


def custom_parser(args):
    cmd_params = load_json(args[1])

    cmd_params["trainer_params"]["use_cuda"] = torch.cuda.is_available()
    cmd_params["num_cpus"] = os.cpu_count()

    cmd_params["loader_params"]["data_dir"] = "data/lavacrossing_shortest_path/S9N1"
    cmd_params["model_dir"] = cmd_params["model_dir"].replace("warcraft_shortest_path", "lavacrossing")
    cmd_params["problem_type"] = "lavacrossing"
    cmd_params["trainer_params"]["neighbourhood_fn"] = "4-grid"

    for arg in args[2:]:
        assert '=' in arg, 'each arg apart from json needs to be in the form a=b'
        key, value = arg.split('=')
        key_entries = key.split('.')

        if value in {'true', 'True'}:
            value = True
        if value in {'false', 'False'}:
            value = False

        sub_param = cmd_params
        for key_entry in key_entries[:-1]:
            if key_entry not in sub_param:
                sub_param[key_entry] = dict()
            sub_param = sub_param[key_entry]

        inferred_type = None
        if key_entries[-1] in sub_param:
            inferred_type = type(sub_param[key_entries[-1]])

        if inferred_type is None:
            print(f'No existing entry for {key}')
            if isinstance(value, str):
                if all(char.isdigit() for char in value):
                    value = int(value)
                elif re.match(r'^-?\d+(?:\.\d+)?$', value) is not None:
                    value = float(value)
            inferred_type = type(value)

        print(f'{sub_param}[{key_entries[-1]}] = {inferred_type}({value})')
        sub_param[key_entries[-1]] = inferred_type(value)

    return cmd_params


def main():
    params = update_params_from_cmdline(verbose=True, custom_parser=custom_parser)
    os.makedirs(params.model_dir, exist_ok=True)
    save_settings_to_json(params, params.model_dir)

    num_cpus = params.get("num_cpus", psutil.cpu_count(logical=True))
    use_ray = params.get("use_ray", False)
    fast_forward_training = params.get("fast_forward_training", False)
    if use_ray:
        ray.init(
            num_cpus=num_cpus,
            logging_level=WARNING,
            ignore_reinit_error=True,
            log_to_driver=False,
            _temp_dir=f'/tmp/{getpass.getuser()}-{os.getpid()}-ray',
            **params.get("ray_params", {})
        )

    set_seed(params.seed)

    Logger.configure(params.model_dir, "tensorboard")

    dataset_loader = dataset_loaders[params.problem_type]
    train_iterator, test_iterator, metadata = dataset_loader(**params.loader_params)

    trainer_class = trainer_loaders[params.problem_type](params.trainer_name)

    fast_mode = params.get("fast_mode", False)
    trainer = trainer_class(
        train_iterator=train_iterator,
        test_iterator=test_iterator,
        metadata=metadata,
        fast_mode=fast_mode,
        **params.trainer_params
    )
    train_results = {}
    for i in range(params.num_epochs):
        if i % params.evaluate_every == 0:
            print('EVALUATING')
            eval_results = trainer.evaluate()
            print(eval_results)

        train_results = trainer.train_epoch()
        if train_results["train_accuracy"] > 0.999 and fast_forward_training:
            print(f'Reached train accuracy of {train_results["train_accuracy"]}. Fast forwarding.')
            break

    print('EVALUATING')
    eval_results = trainer.evaluate()
    print(eval_results)
    train_results = train_results or {}
    save_metrics_params(params=params, metrics={**eval_results, **train_results})

    if use_ray:
        ray.shutdown()


if __name__ == "__main__":
    main()
