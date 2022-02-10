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


def random_translate_plant(plant, global_map, old_pos, thresh, margin):
    # minus 1 for trim_zeros()
    # add 1 back
    # 45 x 49 x 79
    plant = trim_zeros(plant)
    plant_shape_xx, plant_shape_yy, plant_shape_zz = np.shape(plant)
    global_shape_xx, global_shape_yy, global_shape_zz = np.shape(global_map)
    old_x, old_y, old_z = old_pos

    loc_x, loc_y, loc_z = None, None, None

    count = 0

    if plant is not None:
        # make sure the the plant fully fitting within the wall
        max_x = global_shape_xx - plant_shape_xx
        max_y = global_shape_yy - plant_shape_yy

        # randomly initialize the position
        loc_x = random.randint(0, max_x - 1)
        loc_y = random.randint(0, max_y - 1)

        while np.linalg.norm([loc_x - old_x, loc_y - old_y]) < thresh:
            assert margin < max_x - margin and margin < max_y - margin, "Margin is too large"
            loc_x = random.randint(margin, max_x - margin)
            loc_y = random.randint(margin, max_y - margin)
            count += 1
            if count >= 1000:
                logging.info("Loop too many times when generate the plants")
                break
        global_map[loc_x: loc_x + plant_shape_xx, loc_y:loc_y + plant_shape_yy, 0:0 + plant_shape_zz] = plant

    return global_map, (loc_x, loc_y, 0), (loc_x + plant_shape_xx, loc_y + plant_shape_yy, 0 + plant_shape_zz)


def get_random_multi_plant_models(global_map, plants, thresh, margin):
    start_pos = (0., 0., 0.)
    bound_boxes = []

    for plant in plants:
        global_map, start_pos, end_pos = random_translate_plant(plant, global_map, start_pos, thresh, margin)
        bound_boxes.append([start_pos, end_pos])

    return global_map, np.array(bound_boxes)
