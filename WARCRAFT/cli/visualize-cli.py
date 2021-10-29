#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from maprop.decorators import input_to_numpy
from maprop.blackbox.dijkstra import dijkstra
from maprop.utils import concat_2d

from tqdm import tqdm


@input_to_numpy
def draw_paths_on_image(image, true_path, suggested_path, scaling_factor):
    transpose = True
    if len(image.shape) == 3 and image.shape[2] == 3:
        transpose = False
    # print(transpose)

    image = preprocess_image(image, scaling_factor, transpose=transpose)

    true_transitions, is_valid_shortest_path_true = get_transitions_from_path(true_path)

    # sug_transitions, is_valid_shortest_path_sug = get_transitions_from_path(suggested_path)

    visualized = draw_paths_on_image_as_line(image=image,
                                             transitions=true_transitions,
                                             grid_shape=true_path.shape, sf=scaling_factor,
                                             color="#8fb032")

    # visualized = draw_paths_on_image_as_dots(image=visualized,
    #                                          path=suggested_path,
    #                                          sf=scaling_factor, color="#e19c24")

    if transpose:
        visualized = postprocess_image(visualized)

    return visualized


def get_transitions_from_path(path):
    inverted_path = 1.-path
    shortest_path, _, transitions = dijkstra(inverted_path, request_transitions=True)
    is_valid_shortest_path = np.min(shortest_path == path)
    return transitions, is_valid_shortest_path


def draw_paths_on_image_as_line(image, transitions, grid_shape, sf, color):
    im_width, im_height = image.size
    draw = ImageDraw.Draw(image)
    grid_x_max, grid_y_max = grid_shape

    cur_x, cur_y = grid_x_max - 1, grid_y_max - 1
    while (cur_x, cur_y) != (0, 0):
        next_y, next_x = transitions[(cur_y, cur_x)]
        cur_x_im, cur_y_im, _, _ = grid_to_im_coordinate(
            cur_x, cur_y, grid_x_max, grid_y_max, im_width, im_height
        )
        next_x_im, next_y_im, _, _ = grid_to_im_coordinate(
            next_x, next_y, grid_x_max, grid_y_max, im_width, im_height
        )
        draw.line([(cur_x_im, cur_y_im), (next_x_im, next_y_im)],
            fill=color, width=sf)
        cur_x, cur_y = next_x, next_y
    return image


def draw_paths_on_image_as_dots(image, path, sf, color):
    im_width, im_height = image.size
    draw = ImageDraw.Draw(image)
    grid_x_max, grid_y_max = path.shape

    for x, y in np.ndindex(path.shape):
        if path[y][x]:
            x_im, y_im, x_spacing, y_spacing = grid_to_im_coordinate(
                x, y, grid_x_max, grid_y_max, im_width, im_height
            )
            disp = min(x_spacing, y_spacing) // 8
            draw.ellipse([(x_im - disp, y_im - disp), (x_im + disp, y_im + disp)],
                outline=color, width=sf)
    return image


def grid_to_im_coordinate(grid_x, grid_y, grid_x_max, grid_y_max, im_width, im_height):
    x_spacing = im_width // (grid_x_max)
    im_x = x_spacing * grid_x + x_spacing // 2
    y_spacing = im_height // (grid_y_max)
    im_y = y_spacing * grid_y + y_spacing // 2
    return im_x, im_y, x_spacing, y_spacing


def preprocess_image(image, scaling_factor, transpose):
    if len(image.shape) == 5:  # grid of images
        image = concat_2d(image)
    if transpose:
        image = np.moveaxis(image, 0, 2)
    im = Image.fromarray(image, mode=None)
    # im = im.resize(tuple(scaling_factor * x for x in im.size), resample=Image.NEAREST, box=None)
    return im


def postprocess_image(image):
    image = np.moveaxis(np.array(image), 2, 0)
    return image


def main(argv):
    data_dir = 'data/warcraft_shortest_path/30x30/'
    train_prefix = 'train'

    train_labels = np.load(os.path.join(data_dir, train_prefix + "_shortest_paths.npy"))
    train_true_weights = np.load(os.path.join(data_dir, train_prefix + "_vertex_weights.npy"))
    train_maps = np.load(os.path.join(data_dir, train_prefix + "_maps.npy"))

    nb_images = train_maps.shape[0]
    nb_images = 100

    print(train_maps.shape)
    print(train_labels.shape)
    print(train_true_weights.shape)

    for i in tqdm(range(nb_images)):
        maps = train_maps[i]

        # img = preprocess_image(maps, None, False)
        img = draw_paths_on_image(maps, train_labels[i], None, 5)

        img.save(f'images/warcraft/{i}.pdf', dpi=(1000, 1000))
        # img.show()


if __name__ == '__main__':
    main(sys.argv[1:])
