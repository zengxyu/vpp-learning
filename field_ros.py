#!/usr/bin/env python

import numpy as np
from enum import IntEnum
from scipy.spatial.transform import Rotation
import time

from scripts.vpp_env_client import EnvironmentClient


class Action(IntEnum):
    DO_NOTHING = 0
    MOVE_FORWARD = 1
    MOVE_BACKWARD = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    MOVE_UP = 5
    MOVE_DOWN = 6
    ROTATE_ROLL_P = 7
    ROTATE_ROLL_N = 8
    ROTATE_PITCH_P = 9
    ROTATE_PITCH_N = 10
    ROTATE_YAW_P = 11
    ROTATE_YAW_N = 12


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
    def __init__(self, shape, sensor_range, hfov, vfov, max_steps, init_file=None, headless=False, scale=0.05):
        self.found_targets = 0
        self.free_cells = 0
        self.sensor_range = sensor_range
        self.hfov = hfov
        self.vfov = vfov
        self.shape = shape
        self.global_map = np.zeros(self.shape)
        self.known_map = np.zeros(self.shape)
        self.max_steps = max_steps
        self.headless = headless
        self.robot_pos = [0.0, 0.0, 0.0]
        self.robot_rot = Rotation.from_quat([0, 0, 0, 1])
        self.MOVE_STEP = 1.0
        self.ROT_STEP = 15.0

        self.reset_count = 0
        self.upper_scale = 1
        self.ratio = 0.1
        self.client = EnvironmentClient()

        print("max steps:", self.max_steps)
        print("move step:", self.MOVE_STEP)
        print("rot step:", self.ROT_STEP)

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
        relative_move = np.array([0, 0, 0])
        relative_rot = np.array([0, 0, 0, 0])
        if action == Action.MOVE_FORWARD:
            relative_move = np.array([0.1, 0, 0])
        elif action == Action.MOVE_BACKWARD:
            relative_move = np.array([-0.1, 0, 0])
        elif action == Action.MOVE_LEFT:
            relative_move = np.array([0, 0.1, 0])
        elif action == Action.MOVE_RIGHT:
            relative_move = np.array([0, -0.1, 0])
        elif action == Action.MOVE_UP:
            relative_move = np.array([0, 0, 0.1])
        elif action == Action.MOVE_DOWN:
            relative_move = np.array([0, 0, -0.1])
        elif action == Action.ROTATE_ROLL_P:
            r = Rotation.from_euler('x', self.ROT_STEP, degrees=True)
            relative_rot = r.as_quat()
        elif action == Action.ROTATE_ROLL_N:
            r = Rotation.from_euler('x', -self.ROT_STEP, degrees=True)
            relative_rot = r.as_quat()
        elif action == Action.ROTATE_PITCH_P:
            r = Rotation.from_euler('y', self.ROT_STEP, degrees=True)
            relative_rot = r.as_quat()
        elif action == Action.ROTATE_PITCH_N:
            r = Rotation.from_euler('y', -self.ROT_STEP, degrees=True)
            relative_rot = r.as_quat()
        elif action == Action.ROTATE_YAW_N:
            r = Rotation.from_euler('z', -self.ROT_STEP, degrees=True)
            relative_rot = r.as_quat()
        elif action == Action.ROTATE_YAW_P:
            r = Rotation.from_euler('z', self.ROT_STEP, degrees=True)
            relative_rot = r.as_quat()
        relative_pose = np.append(relative_move, relative_rot).tolist()
        start_time = time.time()
        unknownCount, freeCount, occupiedCount, roiCount, robotPose, robotJoints, reward = self.client.sendRelativePose(
            relative_pose)
        # print("robot pose computed by me:{}".format(self.robot_pos))
        self.robot_pos = robotPose[:3]
        # print("robot pose computed by remote:{}".format(self.robot_pos))
        self.robot_rot = Rotation.from_quat(robotPose[3:])
        print("sendRelativeTime:{}".format(time.time() - start_time))
        self.found_targets += reward
        self.step_count += 1
        done = self.step_count >= self.max_steps

        # unknown_map, known_free_map, known_target_map = self.generate_unknown_map(cam_pos)
        map = np.concatenate([unknownCount, freeCount, roiCount], axis=0)

        # return map, np.concatenate(
        #     (self.robot_pos, self.robot_rot.as_quat())), reward, 0, done
        return map, robotPose, reward, done

    def reset(self):
        self.reset_count += 1
        self.known_map = np.zeros(self.shape)

        self.step_count = 0
        self.found_targets = 0
        self.free_cells = 0

        unknownCount, freeCount, occupiedCount, roiCount, robotPose, robotJoints, reward = self.client.sendReset()
        map = np.concatenate([unknownCount, freeCount, roiCount], axis=0)
        return map, robotPose
