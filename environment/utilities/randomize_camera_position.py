#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/22/22 11:17 AM 
    @Description    :
        
===========================================
"""
from typing import List

import numpy as np

from environment.utilities.bound_helper import BoundBox, in_plant_bounds


def randomize_camera_position(sensor_position_bound, bounding_boxes):
    robot_pos = np.random.randint(sensor_position_bound.lower_bound,
                                  sensor_position_bound.upper_bound, size=(3,))
    while in_plant_bounds(bounding_boxes, robot_pos):
        robot_pos = np.random.randint(sensor_position_bound.lower_bound,
                                      sensor_position_bound.upper_bound, size=(3,))
    print("randomized sensor starting point = ", robot_pos)
    return robot_pos
