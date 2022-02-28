#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/28/22 12:41 AM 
    @Description    :
        
===========================================
"""
import numpy as np


class CircularMotionAgent:
    def __init__(self, parser_args, action_space):
        self.action_space = action_space
        self.env_config = parser_args.env_config

        upper_bound_from = np.array(
            [self.env_config["upper_bound_range_x"][0],
             self.env_config["upper_bound_range_y"][0],
             self.env_config["upper_bound_range_z"][0]])

        self.shape = upper_bound_from
        self.plant_position_margin = self.env_config["plant_position_margin"]
        self.step_count = 0
        self.actions = [0, 8]

    def act(self):
        self.step_count += 1
        action = self.actions[self.step_count % len(self.actions)]
        return action
