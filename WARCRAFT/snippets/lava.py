#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy

from gym_minigrid.wrappers import *

import matplotlib.pyplot as plt

numpy.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)


def display(image):
    plt.imshow(image)
    plt.show()


def main():
    env = gym.make('MiniGrid-LavaCrossingS11N5-v0')
    fo_env = FullyObsWrapper(env)
    rgb_env = RGBImgObsWrapper(env)

    fo_obs, _, _, _ = fo_env.step(0)
    rgb_obs, _, _, _ = rgb_env.step(0)

    print(fo_obs['image'].sum(2).T)

    # Get pixel observations
    display(rgb_obs['image'])

    # This now produces an RGB tensor only
    # obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
