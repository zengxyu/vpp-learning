from enum import IntEnum, Enum
import numpy as np
from scipy.spatial.transform import Rotation


class ActionMoRo10(object):
    def __init__(self):
        self.n = 10

    def get_relative_move_rot(self, axes, action, move_step, rot_step):
        relative_move = np.array([0, 0, 0])
        relative_rot = Rotation.from_quat(np.array([0, 0, 0, 1.0]))
        if action == ActionMoRo10IntEnum.MOVE_FORWARD:
            relative_move = axes[0] * move_step
        elif action == ActionMoRo10IntEnum.MOVE_BACKWARD:
            relative_move = -axes[0] * move_step
        elif action == ActionMoRo10IntEnum.MOVE_LEFT:
            relative_move = axes[1] * move_step
        elif action == ActionMoRo10IntEnum.MOVE_RIGHT:
            relative_move = -axes[1] * move_step
        elif action == ActionMoRo10IntEnum.MOVE_UP:
            relative_move = axes[2] * move_step
        elif action == ActionMoRo10IntEnum.MOVE_DOWN:
            relative_move = -axes[2] * move_step
        elif action == ActionMoRo10IntEnum.ROTATE_PITCH_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step) * axes[1])
        elif action == ActionMoRo10IntEnum.ROTATE_PITCH_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step) * axes[1])
        elif action == ActionMoRo10IntEnum.ROTATE_YAW_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step) * axes[2])
        elif action == ActionMoRo10IntEnum.ROTATE_YAW_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step) * axes[2])
        else:
            raise NotImplementedError
        return relative_move, relative_rot


class ActionMoRo20(object):

    def __init__(self):
        self.n = 20
        self.action_space = []
        for mtplier in Multiplier2Enum:
            for action in ActionMoRo10IntEnum:
                self.action_space.append([action, mtplier])

    def get_relative_move_rot(self, axes, action_ind, move_step, rot_step):
        relative_move = np.array([0, 0, 0])
        relative_rot = Rotation.from_quat(np.array([0, 0, 0, 1.0]))
        action, multiplier = self.action_space[action_ind]

        if action == ActionMoRo10IntEnum.MOVE_FORWARD:
            relative_move = axes[0] * move_step * multiplier
        elif action == ActionMoRo10IntEnum.MOVE_BACKWARD:
            relative_move = -axes[0] * move_step * multiplier
        elif action == ActionMoRo10IntEnum.MOVE_LEFT:
            relative_move = axes[1] * move_step * multiplier
        elif action == ActionMoRo10IntEnum.MOVE_RIGHT:
            relative_move = -axes[1] * move_step * multiplier
        elif action == ActionMoRo10IntEnum.MOVE_UP:
            relative_move = axes[2] * move_step * multiplier
        elif action == ActionMoRo10IntEnum.MOVE_DOWN:
            relative_move = -axes[2] * move_step * multiplier
        elif action == ActionMoRo10IntEnum.ROTATE_PITCH_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step * multiplier) * axes[1])
        elif action == ActionMoRo10IntEnum.ROTATE_PITCH_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step * multiplier) * axes[1])
        elif action == ActionMoRo10IntEnum.ROTATE_YAW_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step * multiplier) * axes[2])
        elif action == ActionMoRo10IntEnum.ROTATE_YAW_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step * multiplier) * axes[2])
        else:
            raise NotImplementedError
        return relative_move, relative_rot


class ActionMoRo30(object):

    def __init__(self):
        self.n = 30
        self.action_space = []
        for mtplier in Multiplier3Enum:
            for action in ActionMoRo10IntEnum:
                self.action_space.append([action, mtplier])

    def get_relative_move_rot(self, axes, action_ind, move_step, rot_step):
        relative_move = np.array([0, 0, 0])
        relative_rot = Rotation.from_quat(np.array([0, 0, 0, 1.0]))
        action, multiplier = self.action_space[action_ind]

        if action == ActionMoRo10IntEnum.MOVE_FORWARD:
            relative_move = axes[0] * move_step * multiplier
        elif action == ActionMoRo10IntEnum.MOVE_BACKWARD:
            relative_move = -axes[0] * move_step * multiplier
        elif action == ActionMoRo10IntEnum.MOVE_LEFT:
            relative_move = axes[1] * move_step * multiplier
        elif action == ActionMoRo10IntEnum.MOVE_RIGHT:
            relative_move = -axes[1] * move_step * multiplier
        elif action == ActionMoRo10IntEnum.MOVE_UP:
            relative_move = axes[2] * move_step * multiplier
        elif action == ActionMoRo10IntEnum.MOVE_DOWN:
            relative_move = -axes[2] * move_step * multiplier
        elif action == ActionMoRo10IntEnum.ROTATE_PITCH_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step * multiplier) * axes[1])
        elif action == ActionMoRo10IntEnum.ROTATE_PITCH_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step * multiplier) * axes[1])
        elif action == ActionMoRo10IntEnum.ROTATE_YAW_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step * multiplier) * axes[2])
        elif action == ActionMoRo10IntEnum.ROTATE_YAW_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step * multiplier) * axes[2])
        else:
            raise NotImplementedError
        return relative_move, relative_rot


class ActionMoRo10IntEnum(IntEnum):
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    MOVE_UP = 4
    MOVE_DOWN = 5
    ROTATE_PITCH_P = 6
    ROTATE_PITCH_N = 7
    ROTATE_YAW_P = 8
    ROTATE_YAW_N = 9


Multiplier2Enum = [0.5, 2.5]

Multiplier3Enum = [0.1, 1, 2]


class ActionMo6IntEnum(IntEnum):
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    MOVE_UP = 4
    MOVE_DOWN = 5


class ActionRo6IntEnum(IntEnum):
    ROTATE_ROLL_P = 0
    ROTATE_ROLL_N = 1
    ROTATE_PITCH_P = 2
    ROTATE_PITCH_N = 3
    ROTATE_YAW_P = 4
    ROTATE_YAW_N = 5
