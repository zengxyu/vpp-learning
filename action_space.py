from enum import IntEnum
import numpy as np
from scipy.spatial.transform import Rotation


class ActionMoRo12(IntEnum):
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

    @staticmethod
    def get_relative_move_rot(axes, action, move_step, rot_step):
        relative_move = np.array([0, 0, 0])
        relative_rot = np.array([0, 0, 0, 1.0])
        if action == ActionMoRo12.MOVE_FORWARD:
            relative_move = np.array([1.0, 0, 0]) * move_step
        elif action == ActionMoRo12.MOVE_BACKWARD:
            relative_move = np.array([-1.0, 0, 0]) * move_step
        elif action == ActionMoRo12.MOVE_LEFT:
            relative_move = np.array([0, 1.0, 0]) * move_step
        elif action == ActionMoRo12.MOVE_RIGHT:
            relative_move = np.array([0, -1.0, 0]) * move_step
        elif action == ActionMoRo12.MOVE_UP:
            relative_move = np.array([0, 0, 1.0]) * move_step
        elif action == ActionMoRo12.MOVE_DOWN:
            relative_move = np.array([0, 0, -1.0]) * move_step
        elif action == ActionMoRo12.ROTATE_ROLL_P:
            r = Rotation.from_euler('x', rot_step, degrees=True)
            relative_rot = r.as_quat()
        elif action == ActionMoRo12.ROTATE_ROLL_N:
            r = Rotation.from_euler('x', -rot_step, degrees=True)
            relative_rot = r.as_quat()
        elif action == ActionMoRo12.ROTATE_PITCH_P:
            r = Rotation.from_euler('y', rot_step, degrees=True)
            relative_rot = r.as_quat()
        elif action == ActionMoRo12.ROTATE_PITCH_N:
            r = Rotation.from_euler('y', -rot_step, degrees=True)
            relative_rot = r.as_quat()
        elif action == ActionMoRo12.ROTATE_YAW_P:
            r = Rotation.from_euler('z', rot_step, degrees=True)
            relative_rot = r.as_quat()
        elif action == ActionMoRo12.ROTATE_YAW_N:
            r = Rotation.from_euler('z', -rot_step, degrees=True)
            relative_rot = r.as_quat()
        return relative_move, relative_rot

    @staticmethod
    def get_relative_move_rot2(axes, action, move_step, rot_step):
        relative_move = np.array([0, 0, 0])
        relative_rot = Rotation.from_quat(np.array([0, 0, 0, 1.0]))
        if action == ActionMoRo12.MOVE_FORWARD:
            relative_move = axes[0] * move_step
        elif action == ActionMoRo12.MOVE_BACKWARD:
            relative_move = -axes[0] * move_step
        elif action == ActionMoRo12.MOVE_LEFT:
            relative_move = axes[1] * move_step
        elif action == ActionMoRo12.MOVE_RIGHT:
            relative_move = -axes[1] * move_step
        elif action == ActionMoRo12.MOVE_UP:
            relative_move = axes[2] * move_step
        elif action == ActionMoRo12.MOVE_DOWN:
            relative_move = -axes[2] * move_step
        elif action == ActionMoRo12.ROTATE_ROLL_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step) * axes[0])
        elif action == ActionMoRo12.ROTATE_ROLL_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step) * axes[0])
        elif action == ActionMoRo12.ROTATE_PITCH_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step) * axes[1])
        elif action == ActionMoRo12.ROTATE_PITCH_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step) * axes[1])
        elif action == ActionMoRo12.ROTATE_YAW_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step) * axes[2])
        elif action == ActionMoRo12.ROTATE_YAW_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step) * axes[2])
        else:
            raise NotImplementedError
        return relative_move, relative_rot

    @staticmethod
    def get_action_size():
        return 12


# class ActionMoRoMultiplier36(IntEnum):
#
#     def __init__(self, a, b):
#         self.a = a
#         self.step = b
#
#     def build_actions():
#         action_space = []
#         for action in Action:


class Multiplier(IntEnum):
    LARGE_STEP = 0
    MEDIUM_STEP = 1
    SMALL_STEP = 2


class ActionMo6(IntEnum):
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    MOVE_UP = 4
    MOVE_DOWN = 5


class ActionRo6(IntEnum):
    ROTATE_ROLL_P = 0
    ROTATE_ROLL_N = 1
    ROTATE_PITCH_P = 2
    ROTATE_PITCH_N = 3
    ROTATE_YAW_P = 4
    ROTATE_YAW_N = 5

# class Action2:
#     def __init__(self, a, b):
#         self.a = a
#         self.step = b
#
#     def build_actions():
#         action_space = []
#         for action in Action:
