#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def main():
    mode_type_lst = [1, 2, 3]
    nb_samples_lst = [1, 10, 100, 1000]
    temperature_lst = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

    cmd_lst = []

    for mode_type in mode_type_lst:
        for nb_samples in nb_samples_lst:
            for temperature in temperature_lst:
                cmd = f'PYTHONPATH=. python3 ./cli/warcraft-cli.py settings/warcraft_shortest_path/12x12_map.json ' \
                      f'trainer_params.mode.type={mode_type} ' \
                      f'trainer_params.mode.use_marginal=true ' \
                      f'trainer_params.mode.nb_samples={nb_samples} ' \
                      f'trainer_params.mode.temperature={temperature} ' \
                      f'> logs/warcraft/temperature_v1/warcraft_{mode_type}_{nb_samples}_{temperature}.log 2>&1'
                cmd_lst += [cmd]

    for i, cmd in enumerate(cmd_lst):
        print(f'CUDA_VISIBLE_DEVICES={(i % 3) + 1} {cmd}')


if __name__ == "__main__":
    main()
