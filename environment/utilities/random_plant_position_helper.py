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
from typing import List

import numpy as np

from environment.utilities.bound_helper import BoundBox
from environment.utilities.observable_cells_helper import compute_observable_occ_roi_cells


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


def random_translate_plant(plant, global_map, old_pos_list, thresh, margin: np.array):
    # compute_observable_occ_roi_cells(plant)
    # minus 1 for trim_zeros()
    # add 1 back
    # 45 x 49 x 79
    margin_x, margin_y = margin
    plant = trim_zeros(plant)

    # rotate 90
    if random.random() > 0.5:
        plant = np.transpose(plant, (1, 0, 2))

    # flip
    if random.random() > 0.5:
        plant = np.flip(plant, 1)

    if random.random() > 0.5:
        plant = np.flip(plant, 0)

    plant_shape_xx, plant_shape_yy, plant_shape_zz = np.shape(plant)
    global_shape_xx, global_shape_yy, global_shape_zz = np.shape(global_map)
    # old_x, old_y, old_z = old_pos

    loc_x, loc_y, loc_z = None, None, None

    count = 0

    if plant is not None:
        # make sure the the plant fully fitting within the wall
        max_x = global_shape_xx - plant_shape_xx
        max_y = global_shape_yy - plant_shape_yy

        # randomly initialize the position
        loc_x = random.randint(margin_x, max_x - margin_x)
        loc_y = random.randint(margin_y, max_y - margin_y)

        while check_interval_in_thresh(old_pos_list, [loc_x, loc_y], thresh):
            assert margin_x < max_x - margin_x and margin_y < max_y - margin_y, "Margin is too large"
            loc_x = random.randint(margin_x, max_x - margin_x)
            loc_y = random.randint(margin_y, max_y - margin_y)
            count += 1
            if count >= 1000:
                logging.info("Loop too many times when generate the plants")
                break
        global_map[loc_x: loc_x + plant_shape_xx, loc_y:loc_y + plant_shape_yy, 0:0 + plant_shape_zz] = plant

    return global_map, [loc_x, loc_y, 0], [loc_x + plant_shape_xx, loc_y + plant_shape_yy, 0 + plant_shape_zz]


def check_interval_in_thresh(start_pos_list, pos, thresh):
    if len(start_pos_list) == 0:
        return True
    diffs = np.array(start_pos_list)[:, :2] - pos
    intervals = []
    for diff in diffs:
        interval = np.linalg.norm(diff)
        intervals.append(interval)
    less_thresh = np.array(intervals) < thresh
    flag = any(less_thresh)
    # print("intervals:{}; flag:{}".format(intervals, flag))
    return flag


def get_random_multi_plant_models(global_map: np.array, plants: List[np.array], thresh: float, margin: np.array):
    start_pos_list = []
    bound_boxes = []

    for plant in plants:
        global_map, start_pos, end_pos = random_translate_plant(plant, global_map, start_pos_list, thresh, margin)
        box = BoundBox(lower_bound=start_pos, upper_bound=end_pos)
        bound_boxes.append(box)
        print("box : {}".format(box.__str__()))
        start_pos_list.append(start_pos)

    return global_map, bound_boxes
