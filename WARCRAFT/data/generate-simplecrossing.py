#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Run as:
# cd data/
# PYTHONPATH=.. python3 generate-simplecrossing.py


import os
import sys
import numpy as np

from tqdm import tqdm

from gym_minigrid.wrappers import gym, FullyObsWrapper, RGBImgObsWrapper

from maprop.blackbox.dijkstra import dijkstra

import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)


def display(image):
    plt.imshow(image)
    plt.show()


def generate(env):
    fo_env = FullyObsWrapper(env)
    rgb_env = RGBImgObsWrapper(env)

    fo_obs, _, _, _ = fo_env.step(0)
    rgb_obs, _, _, _ = rgb_env.step(0)

    # Get pixel observations
    # display(rgb_obs['image'])

    matrix = fo_obs['image'][:, :, 0].T
    rgb_matrix = rgb_obs['image']

    matrix = np.where(matrix == 10, 1, matrix)
    matrix = np.where(matrix == 8, 1, matrix)
    matrix = np.where(matrix == 2, 99, matrix)

    cell_size = rgb_matrix.shape[0] // matrix.shape[0]

    matrix = matrix[1:-1, 1:-1]
    rgb_matrix = rgb_matrix[cell_size:-cell_size, cell_size:-cell_size, :]

    matrix = matrix.astype(np.float)

    sp = dijkstra(matrix, "4-grid")
    shortest_path = sp.shortest_path

    return matrix, rgb_matrix, shortest_path


def main():

    variants = ['S9N1', 'S9N2', 'S9N3', 'S11N5']

    dataset_size_pairs = [
        ('train', 10000),
        ('val', 1000),
        ('test', 1000)
    ]

    for variant in variants:
        env_name = f'MiniGrid-SimpleCrossing{variant}-v0'
        env = gym.make(env_name)

        print(f'Environment: {env_name}')

        prefix = f'simplecrossing_shortest_path/{variant}/'
        os.makedirs(prefix)

        for name, size in dataset_size_pairs:
            matrix_lst = []
            rgb_matrix_lst = []
            shortest_path_lst = []

            print(f'\tGenerating {size} examples for {name} ..')

            for i in tqdm(range(size)):
                env.reset()
                env.seed(42 + i)

                matrix, rgb_matrix, shortest_path = generate(env)

                matrix_lst += [matrix]
                rgb_matrix_lst += [rgb_matrix]
                shortest_path_lst += [shortest_path]

            final_matrix = np.concatenate([np.expand_dims(m, 0) for m in matrix_lst], axis=0)
            final_rgb_matrix = np.concatenate([np.expand_dims(m, 0) for m in rgb_matrix_lst], axis=0)
            final_shortest_path = np.concatenate([np.expand_dims(m, 0) for m in shortest_path_lst], axis=0)

            with open(f'{prefix}/{name}_vertex_weights.npy', 'wb') as f:
                np.save(f, final_matrix)

            with open(f'{prefix}/{name}_maps.npy', 'wb') as f:
                np.save(f, final_rgb_matrix)

            with open(f'{prefix}/{name}_shortest_paths.npy', 'wb') as f:
                np.save(f, final_shortest_path)

        env.close()


if __name__ == "__main__":
    main()
