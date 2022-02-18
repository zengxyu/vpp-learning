#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/18/22 3:11 PM 
    @Description    :
        
===========================================
"""
from typing import List
import numpy as np


class BoundBox:
    def __init__(self, lower_bound: np.array, upper_bound: np.array):
        # [self.lower_bound, self.upper_bound] 闭集
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        assert self.lower_bound[0] < self.upper_bound[0] and \
               self.lower_bound[1] < self.upper_bound[1] and \
               self.lower_bound[2] < self.upper_bound[2]


def initialize_world_shape(env_config, randomize_world_size):
    """
    determine the world shape, randomize the shape from the given range
    """
    upper_bound_from = np.array(
        [env_config["upper_bound_range_x"][0],
         env_config["upper_bound_range_y"][0],
         env_config["upper_bound_range_z"][0]])
    upper_bound_to = np.array(
        [env_config["upper_bound_range_x"][1],
         env_config["upper_bound_range_y"][1],
         env_config["upper_bound_range_z"][1]])
    # if not randomizing the shape,
    # then use the [upper_bound_range_x[0], upper_bound_range_y[0], upper_bound_range_z[0]]
    shape = upper_bound_from
    if randomize_world_size:
        shape = np.random.randint(upper_bound_from, upper_bound_to, (3,))
    return shape


def get_world_bound(shape):
    """
    [world_lower_bound, world_upper_bound] 闭集合
    """
    world_lower_bound = np.array([0, 0, 0])
    world_upper_bound = shape - 1
    box = BoundBox(lower_bound=world_lower_bound, upper_bound=world_upper_bound)
    return box


def get_sensor_position_bound(world_bounding_box: BoundBox, sensor_position_margin: np.array):
    """
    get the bound of sensor position, that is the sensor can only move in this bound
    """
    sensor_position_lower_bound = world_bounding_box.lower_bound + sensor_position_margin
    sensor_position_upper_bound = world_bounding_box.upper_bound - sensor_position_margin
    return BoundBox(lower_bound=sensor_position_lower_bound, upper_bound=sensor_position_upper_bound)


def in_plant_bounds(plant_bounding_boxes: List[BoundBox], point: np.array):
    """
    check if point in bounding boxes
    """
    x, y, z = point
    if len(plant_bounding_boxes) != 0:
        for box in plant_bounding_boxes:
            start_x, start_y, start_z = box.lower_bound
            end_x, end_y, end_z = box.upper_bound
            if start_x <= x <= end_x and start_y <= y <= end_y and start_z <= z <= end_z:
                return True
    return False


def out_of_world_bound(world_bounding_box: BoundBox, point: np.array):
    """
    check if point outside of the world
    """
    start_x, start_y, start_z = world_bounding_box.lower_bound
    end_x, end_y, end_z = world_bounding_box.upper_bound
    x, y, z = point
    if (x < start_x or x > end_x) or (y < start_y or y > end_y) or (z < start_z or z > end_z):
        return True
    return False


def out_of_sensor_position_bound(sensor_position_bound: BoundBox, sensor_position: np.array):
    """
    check if the sensor position in the sensor position bound
    """
    lower_bound = sensor_position_bound.lower_bound
    upper_bound = sensor_position_bound.upper_bound
    x, y, z = sensor_position
    if x < lower_bound[0] or x > upper_bound[0] \
            or y < lower_bound[1] or y > upper_bound[1] \
            or z < lower_bound[2] or z > upper_bound[2]:
        return True
    return False
