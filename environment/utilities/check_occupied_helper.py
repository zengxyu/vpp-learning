#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/5/22 4:22 PM 
    @Description    :
        
===========================================
"""
import numpy as np


def has_obstacle(global_map, robot_pos, direction, step=1., partial_direction=[]):
    # if any obstacles in the relative movement,
    length = np.linalg.norm(direction)
    unit_direction = direction / length
    frac = 0.
    while frac < length:
        cur = robot_pos + frac * unit_direction
        x = int(cur[0])
        y = int(cur[1])
        z = int(cur[2])

        if x >= global_map.shape[0] or y >= global_map.shape[2] or z >= global_map.shape[2]:
            partial_direction.append(frac * unit_direction)
            break
        # occupied
        if global_map[x, y, z] == 2:
            partial_direction.append(frac * unit_direction)
            return True

        frac += step
    return False


def in_bound_boxes(bounding_boxes, point):
    """
    check if point in bounding boxes
    """
    x, y, z = point
    if len(bounding_boxes) != 0:
        for start, end in bounding_boxes:
            start_x, start_y, start_z = start
            end_x, end_y, end_z = end
            if start_x <= x <= end_x and start_y <= y <= end_y and start_z <= z <= end_z:
                return True
    return False


def out_of_world(world_bounding_box, point):
    """
    check if point outside of the world
    """
    start, end = world_bounding_box
    start_x, start_y, start_z = start
    end_x, end_y, end_z = end
    x, y, z = point
    if (x < start_x or x >= end_x) or (y < start_y or y >= end_y) or (z < start_z or z >= end_z):
        return True
    return False
