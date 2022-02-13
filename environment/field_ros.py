#!/usr/bin/environment python

import numpy as np
from scipy.spatial.transform import Rotation

from environment.utilities.map_concat_helper import concat
from scripts.vpp_env_client import EnvironmentClient


class FieldRos:
    def __init__(self, parser_args, action_space):
        self.parser_args = parser_args
        self.env_config = parser_args.env_config_ros
        self.training_config = parser_args.training_config
        self.max_steps = self.env_config["max_steps"]
        self.MOVE_STEP = self.env_config["move_step"]
        self.ROT_STEP = self.env_config["rot_step"]

        self.randomize = self.env_config["randomize"]
        self.handle_simulation = self.env_config["handle_simulation"]

        self.action_space = action_space
        self.reset_count = 0

        # following variables need to be reset
        self.robot_pos = [0.0, 0.0, 0.0]
        self.robot_rot = Rotation.from_quat([0, 0, 0, 1])
        self.relative_position = np.array([0., 0., 0.])
        self.relative_rotation = Rotation.from_quat([0, 0, 0, 1])

        self.free_total = 0
        self.occ_total = 0
        self.roi_total = 0

        self.found_roi_sum = 0
        self.found_occ_sum = 0
        self.found_free_sum = 0

        self.step_count = 0

        self.visit_resolution = 16
        self.visit_shape = None
        self.visit_map = None
        self.map = None

        self.client = EnvironmentClient(handle_simulation=self.handle_simulation)
        if self.handle_simulation:
            self.client.startSimulation()

        print("max steps:", self.max_steps)
        print("move step:", self.MOVE_STEP)
        print("rot step:", self.ROT_STEP)

    def initialize(self):
        self.robot_pos = np.array([0.0, 0.0, 0.0])
        self.robot_rot = Rotation.from_quat([0, 0, 0, 1])

        self.relative_position = np.array([0., 0., 0.])
        self.relative_rotation = Rotation.from_quat([0, 0, 0, 1])

        self.step_count = 0
        self.found_roi_sum = 0
        self.found_occ_sum = 0
        self.found_free_sum = 0

    def step(self, action):
        print("step:{}".format(self.step_count))
        axes = self.robot_rot.as_matrix().transpose()
        relative_move, relative_rot = self.action_space.get_relative_move_rot(axes, action, self.MOVE_STEP,
                                                                              self.ROT_STEP)
        relative_pose = np.append(relative_move, relative_rot.as_quat()).tolist()
        unknown_map, known_free_map, known_occupied_map, known_roi_map, robot_pose, \
        found_roi, found_occ, found_free = self.client.sendRelativePose(relative_pose)
        print("robot pose:{}".format(robot_pose))
        robot_pos = np.array(robot_pose[:3])
        robot_rot = Rotation.from_quat(np.array(robot_pose[3:]))

        self.relative_position = robot_pos - self.robot_pos
        relative_rotation, _ = Rotation.align_vectors([robot_rot.as_rotvec()], [self.robot_rot.as_rotvec()])
        self.relative_rotation = relative_rotation

        self.robot_pos = robot_pos
        self.robot_rot = robot_rot

        self.found_roi_sum += found_roi
        self.found_occ_sum += found_occ
        self.found_free_sum += found_free

        self.step_count += 1
        done = self.step_count >= self.max_steps

        self.map = concat(unknown_map, known_free_map, known_roi_map, np.uint8)

        inputs = self.get_inputs()
        reward = self.get_reward(0, found_free, found_occ, found_roi)

        info = {"visit_gain": 0, "new_free_cells": found_free, "new_occupied_cells": found_occ,
                "new_found_rois": found_roi, "reward": reward, "coverage_rate": np.nan}
        return inputs, reward, done, info

    def get_reward(self, visit_gain, found_free, found_occ, found_roi):
        weight = self.training_config["rewards"]
        reward = weight["visit_gain_weight"] * visit_gain + \
                 weight["free_weight"] * found_free + \
                 weight["occ_weight"] * found_occ + \
                 weight["roi_weight"] * found_roi
        return reward

    def get_inputs(self):
        relative_movement = np.append(self.relative_position, self.relative_rotation.as_euler('xyz'))
        absolute_movement = np.append(self.robot_pos, self.robot_rot.as_euler('xyz'))

        # create input
        if self.training_config["input"]["observation_map"] and self.training_config["input"]["absolute_movement"]:
            return self.map, absolute_movement

        if self.training_config["input"]["observation_map"] and self.training_config["input"]["relative_movement"]:
            return self.map, relative_movement

        if self.training_config["input"]["observation_map"] and self.training_config["input"]["visit_map"]:
            return self.map, np.array([self.visit_map])

        if self.training_config["input"]["observation_map"]:
            return np.array(self.map)

        if self.training_config["input"]["visit_map"]:
            return np.array([self.visit_map])


    def reset(self):
        print("-----------------------------------reset or randomize!-------------------------------------------")
        self.reset_count += 1
        self.step_count = 0

        unknown_map, known_free_map, known_occupied_map, known_roi_map, robot_pose, new_roi_cells, new_occupied_cells, new_free_cells = self.client.sendReset(
            randomize=self.randomize, min_point=[-1, -1, -0.1], max_point=[1, 1, 0.1], min_dist=0.4)

        self.map = concat(unknown_map, known_free_map, known_roi_map, np.uint8)
        inputs = self.get_inputs()

        return inputs, {}

    def reset_stuck_env(self):
        self.shutdown_environment()
        self.start_environment()

    def shutdown_environment(self):
        print('-----------------------------------SHUTDOWN--------------------------------')
        self.client.shutdownSimulation()

    def start_environment(self):
        print('-----------------------------------RESTART---------------------------------')
        self.client.startSimulation()
