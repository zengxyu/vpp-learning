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


def random_translate_environment(global_map, global_shape, pos):
    # minus 1 for trim_zeros()
    # add 1 back
    trim_data = trim_zeros(global_map)
    trim_data_shape = np.shape(trim_data)
    old_x, old_y, old_z = pos
    result = None
    if trim_data is not None and trim_data_shape is not None:
        wall = np.zeros(global_shape, dtype=np.int32)
        # make sure the the plant fully fitting within the wall
        loc_max_x, loc_max_y, loc_max_z = global_shape[0] - trim_data_shape[0], \
                                          global_shape[1] - trim_data_shape[1], \
                                          global_shape[2] - trim_data_shape[2]
        # randomly initialize the position
        loc_x = random.randint(0, loc_max_x - 1)
        loc_y = random.randint(0, loc_max_y - 1)
        loc_z = random.randint(0, loc_max_z - 1)
        while np.linalg.norm([loc_x - old_x, loc_y - old_y, loc_z - old_z]) < 100:
            loc_x = random.randint(0, loc_max_x - 1)
            loc_y = random.randint(0, loc_max_y - 1)
            loc_z = random.randint(0, loc_max_z - 1)
        result = paste(wall, trim_data, (loc_x, loc_y, loc_z))
        result = result.astype(int)
    return result, (loc_x, loc_y, loc_z)


def expand_and_randomize_environment(global_map, global_shape):
    trans_map1, pos1 = random_translate_environment(global_map, global_shape, (0, 0, 0))
    trans_map2, pos2 = random_translate_environment(global_map, global_shape, pos1)
    global_map = np.logical_or(trans_map1, trans_map2).astype(int)
    return global_map


# def expand_and_randomize_environment(global_map, global_shape, num_trees):
#     trim_data = trim_zeros(global_map)
#     trim_data_shape = np.shape(trim_data)
#     loc_max_x, loc_max_y, loc_max_z = global_shape[0] - trim_data_shape[0], \
#                                       global_shape[1] - trim_data_shape[1], \
#                                       global_shape[2] - trim_data_shape[2]
#     # randomly initialize the position
#     trans_x = random.randint(0, loc_max_x - 1)
#     trans_y = random.randint(30, loc_max_y - 1)
#     trans_global_map = np.roll(global_map.copy(), shift=trans_y, axis=1)
#     global_map = np.logical_or(global_map, trans_global_map).astype(int)
#
#     trans_z = random.randint(30, loc_max_z - 1)
#     trans_global_map = np.roll(global_map.copy(), shift=trans_z, axis=2)
#     global_map = np.logical_or(global_map, trans_global_map).astype(int)
#
#     return global_map


if __name__ == '__main__':
    arr = np.array([[[1, 1, 0, 0], [1, 1, 0, 0]],
                    [[0, 0, 0, 0], [0, 0, 0, 0]]])
    arr2 = np.roll(arr, 1, axis=0)
    print(arr2)
