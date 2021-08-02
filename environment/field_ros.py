#!/usr/bin/environment python

import numpy as np
from enum import IntEnum
from scipy.spatial.transform import Rotation
import time

from action_space import ActionMoRo12
from scripts.vpp_env_client import EnvironmentClient


class FieldValues(IntEnum):
    UNKNOWN = 0,
    FREE = 1,
    OCCUPIED = 2,
    TARGET = 3


class GuiFieldValues(IntEnum):
    FREE_UNSEEN = 0,
    OCCUPIED_UNSEEN = 1,
    TARGET_UNSEEN = 2,
    FREE_SEEN = 3,
    OCCUPIED_SEEN = 4,
    TARGET_SEEN = 5


class Field:
    def __init__(self, Action, shape, sensor_range, hfov, vfov, max_steps, handle_simulation):
        self.found_targets = 0
        self.free_cells = 0
        self.sensor_range = sensor_range
        self.hfov = hfov
        self.vfov = vfov
        self.shape = shape
        self.action_instance = Action()
        self.global_map = np.zeros(self.shape)
        self.known_map = np.zeros(self.shape)
        self.max_steps = max_steps
        self.robot_pos = [0.0, 0.0, 0.0]
        self.robot_rot = Rotation.from_quat([0, 0, 0, 1])
        self.MOVE_STEP = 0.1
        self.ROT_STEP = 15.0

        self.reset_count = 0
        self.upper_scale = 1
        self.ratio = 0.1
        self.client = EnvironmentClient(handle_simulation=handle_simulation)
        if handle_simulation:
            self.client.startSimulation()

        print("max steps:", self.max_steps)
        print("move step:", self.MOVE_STEP)
        print("rot step:", self.ROT_STEP)

    def get_action_size(self):
        return self.action_instance.get_action_size()

    def move_robot(self, direction):
        self.robot_pos += direction
        # self.robot_pos = np.clip(self.robot_pos, self.allowed_lower_bound, self.allowed_upper_bound)

    def rotate_robot(self, axis, angle):
        rot = Rotation.from_rotvec(np.radians(angle) * axis)
        self.robot_rot = rot * self.robot_rot

    def relative_rotation(self, axis, angle):
        rot = Rotation.from_rotvec(np.radians(angle) * axis)
        return rot.as_quat()

    def step(self, action):
        axes = self.robot_rot.as_matrix().transpose()
        relative_move, relative_rot = self.action_instance.get_relative_move_rot2(axes, action, self.MOVE_STEP,
                                                                                  self.ROT_STEP)
        relative_pose = np.append(relative_move, relative_rot.as_quat()).tolist()
        unknownCount, freeCount, occupiedCount, roiCount, robotPose, robotJoints, reward = self.client.sendRelativePose(
            relative_pose)
        self.found_targets += reward
        self.step_count += 1
        done = self.step_count >= self.max_steps
        map = np.concatenate([unknownCount, freeCount, roiCount], axis=0)

        self.robot_pos = np.array(robotPose[:3])
        self.robot_rot = Rotation.from_quat(robotPose[3:])

        return (map, robotPose), reward, done, {}

    def reset(self):
        print("-----------------------------------reset!-------------------------------------------")
        self.reset_count += 1
        self.known_map = np.zeros(self.shape)

        self.step_count = 0
        self.found_targets = 0
        self.free_cells = 0
        unknownCount, freeCount, occupiedCount, roiCount, robotPose, robotJoints, reward = self.client.sendReset()
        # unknownCount, freeCount, occupiedCount, roiCount, robotPose, robotJoints, reward, totalRoiCells = self.client.sendReset()
        # print("total roi cells:{}".format(totalRoiCells))

        map = np.concatenate([unknownCount, freeCount, roiCount], axis=0)

        return (map, robotPose)

    def reset_and_randomize(self):
        print("-------------------------------reset and randomize!-----------------------------------")
        self.reset_count += 1
        self.known_map = np.zeros(self.shape)

        self.step_count = 0
        self.found_targets = 0
        self.free_cells = 0

        unknownCount, freeCount, occupiedCount, roiCount, robotPose, robotJoints, reward = self.client.sendResetAndRandomize(
            [-1, -1, -0.1], [1, 1, 0.1], 0.4)
        # print("total roi cells:{}".format(totalRoiCells))

        map = np.concatenate([unknownCount, freeCount, roiCount], axis=0)

        return (map, robotPose)

    def shutdown_environment(self):
        print('-----------------------------------SHUTDOWN--------------------------------')
        self.client.shutdownSimulation()

    def start_environment(self):
        print('-----------------------------------RESTART---------------------------------')
        self.client.startSimulation()
