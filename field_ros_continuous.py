#!/usr/bin/env python

import numpy as np
from enum import IntEnum
from scipy.spatial.transform import Rotation
import time

from scripts.vpp_env_client import EnvironmentClient


# class Action(IntEnum):
#     MOVE_FORWARD = 0
#     MOVE_BACKWARD = 1
#     MOVE_LEFT = 2
#     MOVE_RIGHT = 3
#     MOVE_UP = 4
#     MOVE_DOWN = 5
#     ROTATE_ROLL_P = 7
#     ROTATE_ROLL_N = 8
#     ROTATE_PITCH_P = 9
#     ROTATE_PITCH_N = 10
#     ROTATE_YAW_P = 11
#     ROTATE_YAW_N = 12

class Action(IntEnum):
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    MOVE_UP = 4
    MOVE_DOWN = 5
    ROTATE_ROLL_P = 6
    ROTATE_ROLL_N = 7
    ROTATE_PITCH_P = 8
    ROTATE_PITCH_N = 9
    ROTATE_YAW_P = 10
    ROTATE_YAW_N = 11


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
    def __init__(self, shape, sensor_range, hfov, vfov, max_steps, handle_simulation):
        self.found_targets = 0
        self.free_cells = 0
        self.sensor_range = sensor_range
        self.hfov = hfov
        self.vfov = vfov
        self.shape = shape
        self.actions = Action
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
        return len(self.robot_pos) + len(self.robot_rot.as_euler('xyz'))

    def move_robot(self, direction):
        self.robot_pos += direction
        # self.robot_pos = np.clip(self.robot_pos, self.allowed_lower_bound, self.allowed_upper_bound)

    def rotate_robot(self, axis, angle):
        rot = Rotation.from_rotvec(np.radians(angle) * axis)
        self.robot_rot = rot * self.robot_rot

    def rotate_robot_aa(self, angle):
        rot = Rotation.from_euler("xyz", angle)
        self.robot_rot = rot * self.robot_rot

    def step(self, action):
        relative_move = action[:3] * self.MOVE_STEP

        relative_rot = Rotation.from_euler("xyz", action[3:] * self.ROT_STEP).as_quat()

        relative_pose = np.append(relative_move, relative_rot).tolist()
        start_time = time.time()
        unknownCount, freeCount, occupiedCount, roiCount, robotPose, robotJoints, reward = self.client.sendRelativePose(
            relative_pose)
        # print("sendRelativeTime:{}".format(time.time() - start_time))
        self.found_targets += reward
        self.step_count += 1
        done = self.step_count >= self.max_steps
        # unknown_map, known_free_map, known_target_map = self.generate_unknown_map(cam_pos)
        map = np.concatenate([unknownCount, freeCount, roiCount], axis=0)

        # return map, np.concatenate(
        #     (self.robot_pos, self.robot_rot.as_quat())), reward, 0, done
        return map, robotPose, reward, done

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

        return map, robotPose

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

        return map, robotPose

    def shutdown_environment(self):
        print('-----------------------------------SHUTDOWN--------------------------------')
        self.client.shutdownSimulation()

    def start_environment(self):
        print('-----------------------------------RESTART---------------------------------')
        self.client.startSimulation()
