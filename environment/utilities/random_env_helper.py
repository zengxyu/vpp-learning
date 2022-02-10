#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/2/22 4:38 PM 
    @Description    :
        
===========================================
"""
import logging
import random

import numpy as np


def paste_slices(tup):
    pos, w, max_w = tup
    wall_min = max(pos, 0)
    wall_max = min(pos + w, max_w)
    block_min = -min(pos, 0)
    block_max = max_w - max(pos + w, max_w)
    block_max = block_max if block_max != 0 else None
    return slice(wall_min, wall_max), slice(block_min, block_max)


def paste(wall, block, loc):
    if block.shape[0] + loc[0] >= wall.shape[0] or block.shape[1] + loc[1] >= wall.shape[1] or block.shape[2] + loc[
        2] >= wall.shape[2]:
        return None
    loc_zip = zip(loc, block.shape, wall.shape)
    wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
    wall[wall_slices] = block[block_slices]
    return wall


def trim_zeros(arr):
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
    return arr[slices]


def random_translate_environment(global_map, global_shape, old_pos, thresh=300):
    # minus 1 for trim_zeros()
    # add 1 back
    # 45 x 49 x 79
    trim_data = trim_zeros(global_map)
    global_shape_z, global_shape_x, global_shape_y = global_shape
    trim_data_z, trim_data_x, trim_data_y = np.shape(trim_data)
    old_z, old_x, old_y = old_pos

    new_global_map = np.zeros_like(global_map).astype(int)

    loc_z, loc_x, loc_y = None, None, None

    count = 0

    if trim_data is not None:
        # make sure the the plant fully fitting within the wall
        max_z = global_shape_z - trim_data_z
        max_x = global_shape_x - trim_data_x
        max_y = global_shape_y - trim_data_y

        # randomly initialize the position
        loc_z = random.randint(0, max_z - 1)
        loc_x = random.randint(0, max_x - 1)
        loc_y = random.randint(0, max_y - 1)

        while np.linalg.norm([loc_z - old_z, loc_x - old_x, loc_y - old_y]) < thresh:
            loc_z = random.randint(0, max_z - 1)
            loc_x = random.randint(0, max_x - 1)
            loc_y = random.randint(0, max_y - 1)
            count += 1
            if count >= 1000:
                logging.info("Loop too many times when generate the plants")
                break

        new_global_map[loc_z:loc_z + trim_data_z, loc_x:loc_x + trim_data_x, loc_y:loc_y + trim_data_y] = trim_data
        # result = paste(wall, trim_data, (loc_x, loc_y, loc_z))

    return new_global_map, (loc_z, loc_x, loc_y), (loc_z + trim_data_z, loc_x + trim_data_x, loc_y + trim_data_y)


def random_translate_plant(plant, global_map, old_pos, thresh):
    # minus 1 for trim_zeros()
    # add 1 back
    # 45 x 49 x 79
    trim_data = trim_zeros(plant)
    trim_data_z, trim_data_x, trim_data_y = np.shape(trim_data)
    global_shape_z, global_shape_x, global_shape_y = np.shape(global_map)
    old_z, old_x, old_y = old_pos

    loc_z, loc_x, loc_y = None, None, None

    count = 0

    if trim_data is not None:
        # make sure the the plant fully fitting within the wall
        max_z = global_shape_z - trim_data_z
        max_x = global_shape_x - trim_data_x
        max_y = global_shape_y - trim_data_y

        # randomly initialize the position
        loc_x = random.randint(0, max_x - 1)
        loc_y = random.randint(0, max_y - 1)

        while np.linalg.norm([loc_x - old_x, loc_y - old_y]) < thresh:
            # loc_z = random.randint(0, max_z - 1)
            loc_x = random.randint(20, max_x - 20)
            loc_y = random.randint(20, max_y - 20)
            count += 1
            if count >= 1000:
                logging.info("Loop too many times when generate the plants")
                break

        global_map[0: trim_data_z, loc_x:loc_x + trim_data_x, loc_y:loc_y + trim_data_y] = trim_data

    return global_map, (0, loc_x, loc_y), (trim_data_z, loc_x + trim_data_x, loc_y + trim_data_y)


def get_random_multi_tree_environment(global_map, global_shape, num, thresh):
    """
    get random multiple tree environment
    """
    start_pos = (0., 0., 0.)
    bound_boxes = []
    new_global_map = np.zeros_like(global_map).astype(int)

    for i in range(num):
        trans_map, start_pos, end_pos = random_translate_environment(global_map, global_shape, start_pos, thresh)
        bound_boxes.append([start_pos, end_pos])
        new_global_map = np.logical_or(new_global_map, trans_map).astype(int)

    return new_global_map, np.array(bound_boxes)


def get_random_multi_plant_models(global_map, plants, thresh):
    start_pos = (0., 0., 0.)
    bound_boxes = []

    for plant in plants:
        global_map, start_pos, end_pos = random_translate_plant(plant, global_map, start_pos, thresh)
        bound_boxes.append([start_pos, end_pos])
    # assert np.sum(global_map == 1) == np.sum(plants[0] == 1) + np.sum(plants[1] == 1) + np.sum(plants[2] == 1) + np.sum(
    #     plants[3] == 1)
    # assert np.sum(global_map == 2) == np.sum(plants[0] == 2) + np.sum(plants[1] == 2) + np.sum(plants[2] == 2) + np.sum(
    #     plants[3] == 2)
    # assert np.sum(global_map == 3) == np.sum(plants[0] == 3) + np.sum(plants[1] == 3) + np.sum(plants[2] == 3) + np.sum(
    #     plants[3] == 3)

    return global_map, np.array(bound_boxes)


if __name__ == '__main__':
    arr = np.array([[[1, 1, 0, 0], [1, 1, 0, 0]],
                    [[0, 0, 0, 0], [0, 0, 0, 0]]])
    arr2 = np.roll(arr, 1, axis=0)
    print(arr2)
