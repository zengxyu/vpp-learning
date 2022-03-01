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
        self.x_length = self.env_config["upper_bound_range_x"][0]
        self.y_length = self.env_config["upper_bound_range_y"][0]
        self.z_length = self.env_config["upper_bound_range_z"][0]
        self.move_step = self.env_config["move_step"]
        self.rot_step = self.env_config["rot_step"]

        self.x_move_action_num = int(self.x_length / self.move_step)
        self.y_move_action_num = int(self.y_length / self.move_step)

        self.shape = upper_bound_from
        self.plant_position_margin = self.env_config["plant_position_margin"]
        self.step_count = 0
        self.actions = []
        # 3 move right, 8 rotate
        for k in range(2):
            for i in range(self.x_move_action_num):
                self.actions.append(3)
                self.random_rotates()

            for i in range(6):
                self.actions.append(8)

            for j in range(self.y_move_action_num):
                self.actions.append(3)
                self.random_rotates()

            for i in range(6):
                self.actions.append(8)

    def random_rotates(self):
        random_rotates = np.random.randint(6, 10, 2)
        back_rotates = []
        for random_rotate in random_rotates:
            if random_rotate == 6:
                back_rotates.append(7)
            if random_rotate == 7:
                back_rotates.append(6)
            if random_rotate == 8:
                back_rotates.append(9)
            if random_rotate == 9:
                back_rotates.append(8)
        self.actions.extend(random_rotates)
        self.actions.extend(back_rotates[::-1])

    def act(self):
        self.step_count += 1

        action = self.actions[self.step_count % len(self.actions)]
        print(self.step_count)
        if self.step_count >= self.env_config["max_steps"]:
            self.step_count = 0
        return action
