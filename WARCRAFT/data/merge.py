#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

paths = [
    'warcraft_shortest_path/30x30/train_shortest_paths_part1.npy',
    'warcraft_shortest_path/30x30/train_maps_part1.npy',
    'warcraft_shortest_path/30x30/train_vertex_weights_part1.npy',
    'warcraft_shortest_path/18x18/train_shortest_paths_part1.npy',
    'warcraft_shortest_path/18x18/train_maps_part1.npy',
    'warcraft_shortest_path/18x18/train_vertex_weights_part1.npy',
    'warcraft_shortest_path/24x24/train_shortest_paths_part1.npy',
    'warcraft_shortest_path/24x24/train_maps_part1.npy',
    'warcraft_shortest_path/24x24/train_vertex_weights_part1.npy'
]

for path in paths:
    path_a = path.replace("part1", "part0")
    path_b = path
    output_path = path.replace("_part1", "")

    a = np.load(path_a)
    b = np.load(path_b)
    c = np.concatenate([a, b], axis=0)

    print(path_a, path_b, output_path)
    np.save(output_path, c)
