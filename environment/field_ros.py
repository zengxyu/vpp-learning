#!/usr/bin/environment python
import os

import numpy as np
from scipy.spatial.transform import Rotation

from environment.utilities.map_concat_helper import concat
from environment.utilities.rotation_helper import get_rotation_between_rotations
from environment.ros_client.vpp_env_client import EnvironmentClient


class FieldRos:
    def __init__(self, parser_args, action_space):
        self.parser_args = parser_args
        self.env_config = parser_args.env_config_ros
        self.training_config = parser_args.training_config

        self.shape_low_bound = (
            self.env_config["shape_x_range"][0], self.env_config["shape_y_range"][0],
            self.env_config["shape_z_range"][0])
        self.shape_high_bound = (
            self.env_config["shape_x_range"][1], self.env_config["shape_y_range"][1],
            self.env_config["shape_z_range"][1])

        self.shape = (
            self.shape_high_bound[0] - self.shape_low_bound[0],
            self.shape_high_bound[1] - self.shape_low_bound[1],
            self.shape_high_bound[2] - self.shape_low_bound[2])

        self.max_steps = self.env_config["max_steps"]
        self.MOVE_STEP = self.env_config["move_step"]
        self.ROT_STEP = self.env_config["rot_step"]

        self.randomize = self.env_config["randomize"]
        self.handle_simulation = self.env_config["handle_simulation"]
        self.resolution = self.env_config["resolution"]

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

        self.visit_resolution = 8
        self.visit_shape = None
        self.visit_map = None
        self.map = None
        self.stuck_count = 0
        self.collision_count = 0
        self.zero_rois_count = 0
        self.client = EnvironmentClient(self.handle_simulation, self.env_config["world_name"], self.env_config["base"],
                                        self.parser_args.head)
        if self.handle_simulation:
            self.client.startSimulation()

        print("max steps:", self.max_steps)
        print("move step:", self.MOVE_STEP)
        print("rot step:", self.ROT_STEP)

    def process_died(self):
        return self.client.process_died()

    def initialize(self):

        self.relative_position = np.array([0., 0., 0.])
        self.relative_rotation = Rotation.from_quat([0, 0, 0, 1])

        self.step_count = 0
        self.found_roi_sum = 0
        self.found_occ_sum = 0
        self.found_free_sum = 0

        self.visit_shape = (int(self.shape[0] // self.visit_resolution), int(self.shape[1] // self.visit_resolution),
                            int(self.shape[2] // self.visit_resolution))
        self.visit_map = np.zeros(shape=self.visit_shape, dtype=np.uint8)
        self.stuck_count = 0
        self.collision_count = 0
        self.zero_rois_count = 0

    def step(self, action):
        # actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # action = actions[self.step_count % len(actions)]
        print("action:{}".format(action))
        axes = self.robot_rot.as_matrix().transpose()
        relative_move, relative_rot = self.action_space.get_relative_move_rot(axes, action, self.MOVE_STEP,
                                                                              self.ROT_STEP)
        # relative_move_ros =
        relative_pose = np.append(relative_move * self.resolution, relative_rot.as_quat()).tolist()

        unknown_map, known_free_map, known_occupied_map, known_roi_map, robot_pose, \
        found_roi, found_occ, found_free, has_move = self.client.sendRelativePose(relative_pose)
        if self.process_died() or has_move is None:
            print(
                "+++++++++++++++++++++++++++++++++++++NOTICE gazebo Crashed++++++++++++++++++++++++++++++++++++++++++++++")
            return None, None, None, None
        else:
            robot_pos = np.array(robot_pose[:3]) / self.resolution
            robot_rot = Rotation.from_quat(np.array(robot_pose[3:]))

            self.relative_position = robot_pos - self.robot_pos
            self.relative_rotation = get_rotation_between_rotations(self.robot_rot, robot_rot)

            self.robot_pos = robot_pos
            self.robot_rot = robot_rot

            visit_gain, coverage_rate = self.update_visit_map()

            self.found_roi_sum += found_roi
            self.found_occ_sum += found_occ
            self.found_free_sum += found_free

            self.step_count += 1

            self.map = concat(unknown_map, known_free_map, known_occupied_map, known_roi_map, np.uint8)

            collision = not has_move

            if collision:
                self.collision_count += 1
            if found_roi == 0:
                self.zero_rois_count += 1
            else:
                self.zero_rois_count = 0

            inputs = self.get_inputs()
            reward = self.get_reward(visit_gain, found_free, found_occ, found_roi, collision)

            done = self.step_count >= self.max_steps or self.zero_rois_count >= 25

            info = {"visit_gain": visit_gain, "new_free_cells": found_free, "new_occupied_cells": found_occ,
                    "new_found_rois": found_roi, "reward": reward, "coverage_rate": coverage_rate,
                    "collision": collision}
            print("robot pos : {}; robot rotation : {}".format(self.robot_pos, self.robot_rot.as_euler("xyz")))
            print("self.zero_rois_count:{}".format(self.zero_rois_count))

            return inputs, reward, done, info

    def get_reward(self, visit_gain, found_free, found_occ, found_roi, collision):
        weight = self.training_config["rewards"]
        reward = weight["visit_gain_weight"] * visit_gain + \
                 weight["free_weight"] * found_free + \
                 weight["occ_weight"] * found_occ + \
                 weight["roi_weight"] * found_roi + \
                 weight["collision_weight"] * max(self.collision_count - 1, 0)
        print(self.step_count, "collision:{}; reward:{}".format(collision, reward))

        return reward

    def update_visit_map(self):
        # to encourage exploration
        location = np.array([(self.robot_pos[0] - self.shape_low_bound[0]) // self.visit_resolution,
                             (self.robot_pos[1] - self.shape_low_bound[1]) // self.visit_resolution,
                             (self.robot_pos[2] - self.shape_low_bound[2]) // self.visit_resolution]).astype(np.int)
        # neighbor box
        nd = 1
        start_x = max(0, location[0] - nd)
        end_x = min(self.visit_shape[0], location[0] + nd + 1)
        start_y = max(0, location[1] - nd)
        end_y = min(self.visit_shape[1], location[1] + nd + 1)
        start_z = max(0, location[2] - nd)
        end_z = min(self.visit_shape[2], location[2] + nd + 1)

        neighbor_visit_map = self.visit_map[start_x: end_x, start_y: end_y, start_z: end_z]
        visit_gain = np.sum(1 - neighbor_visit_map)
        self.visit_map[start_x: end_x, start_y: end_y, start_z: end_z] = np.ones_like(neighbor_visit_map).astype(
            np.uint8)
        coverage_rate = np.sum(self.visit_map) / np.product(self.visit_shape)
        return visit_gain, coverage_rate

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

        self.initialize()

        # but the limitation from -1 to 1 was mainly for the static arm
        unknown_map, known_free_map, known_occupied_map, known_roi_map, robot_pose, new_roi_cells, new_occupied_cells, new_free_cells, has_move = self.client.sendReset(
            randomize=self.randomize, min_point=[-1, -1, -0.1], max_point=[1, 1, 0.1], min_dist=0.4)
        if self.process_died() or has_move is None:
            print(
                "+++++++++++++++++++++++++++++++++++++NOTICE gazebo Crashed++++++++++++++++++++++++++++++++++++++++++++++")
            return None, None
        else:
            self.robot_pos = np.array(robot_pose[:3]) / self.resolution
            print("======================initialized robot pose:{}====================".format(self.robot_pos))
            self.robot_rot = Rotation.from_quat(np.array(robot_pose[3:]))
            self.map = concat(unknown_map, known_free_map, known_occupied_map, known_roi_map, np.uint8)
            inputs = self.get_inputs()

            return inputs, {}

    def reset_stuck_env(self):
        if self.handle_simulation:
            self.client.socket.close()
            self.shutdown_environment()
            os.system("killall -9 gzserver")
            os.system("killall -9 gzclient")
            self.client = EnvironmentClient(self.handle_simulation, self.env_config["world_name"],
                                            self.env_config["base"],
                                            self.parser_args.head)

            self.start_environment()

    def shutdown_environment(self):
        print('-----------------------------------------------------SHUTDOWN-----------------------------------------')
        self.client.shutdownSimulation()

    def start_environment(self):
        print('-------------------------------------------------RESTART----------------------------------------------')
        self.client.startSimulation()
