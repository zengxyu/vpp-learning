from enum import IntEnum
import numpy as np
from scipy.spatial.transform import Rotation


class ActionMoRo12(object):

    def get_relative_move_rot(self, axes, action, move_step, rot_step):
        relative_move = np.array([0, 0, 0])
        relative_rot = np.array([0, 0, 0, 1.0])
        if action == ActionMoRo12IntEnum.MOVE_FORWARD:
            relative_move = np.array([1.0, 0, 0]) * move_step
        elif action == ActionMoRo12IntEnum.MOVE_BACKWARD:
            relative_move = np.array([-1.0, 0, 0]) * move_step
        elif action == ActionMoRo12IntEnum.MOVE_LEFT:
            relative_move = np.array([0, 1.0, 0]) * move_step
        elif action == ActionMoRo12IntEnum.MOVE_RIGHT:
            relative_move = np.array([0, -1.0, 0]) * move_step
        elif action == ActionMoRo12IntEnum.MOVE_UP:
            relative_move = np.array([0, 0, 1.0]) * move_step
        elif action == ActionMoRo12IntEnum.MOVE_DOWN:
            relative_move = np.array([0, 0, -1.0]) * move_step
        elif action == ActionMoRo12IntEnum.ROTATE_ROLL_P:
            r = Rotation.from_euler('x', rot_step, degrees=True)
            relative_rot = r.as_quat()
        elif action == ActionMoRo12IntEnum.ROTATE_ROLL_N:
            r = Rotation.from_euler('x', -rot_step, degrees=True)
            relative_rot = r.as_quat()
        elif action == ActionMoRo12IntEnum.ROTATE_PITCH_P:
            r = Rotation.from_euler('y', rot_step, degrees=True)
            relative_rot = r.as_quat()
        elif action == ActionMoRo12IntEnum.ROTATE_PITCH_N:
            r = Rotation.from_euler('y', -rot_step, degrees=True)
            relative_rot = r.as_quat()
        elif action == ActionMoRo12IntEnum.ROTATE_YAW_P:
            r = Rotation.from_euler('z', rot_step, degrees=True)
            relative_rot = r.as_quat()
        elif action == ActionMoRo12IntEnum.ROTATE_YAW_N:
            r = Rotation.from_euler('z', -rot_step, degrees=True)
            relative_rot = r.as_quat()
        return relative_move, relative_rot

    def get_relative_move_rot2(self, axes, action, move_step, rot_step):
        relative_move = np.array([0, 0, 0])
        relative_rot = Rotation.from_quat(np.array([0, 0, 0, 1.0]))
        if action == ActionMoRo12IntEnum.MOVE_FORWARD:
            relative_move = axes[0] * move_step
        elif action == ActionMoRo12IntEnum.MOVE_BACKWARD:
            relative_move = -axes[0] * move_step
        elif action == ActionMoRo12IntEnum.MOVE_LEFT:
            relative_move = axes[1] * move_step
        elif action == ActionMoRo12IntEnum.MOVE_RIGHT:
            relative_move = -axes[1] * move_step
        elif action == ActionMoRo12IntEnum.MOVE_UP:
            relative_move = axes[2] * move_step
        elif action == ActionMoRo12IntEnum.MOVE_DOWN:
            relative_move = -axes[2] * move_step
        elif action == ActionMoRo12IntEnum.ROTATE_ROLL_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step) * axes[0])
        elif action == ActionMoRo12IntEnum.ROTATE_ROLL_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step) * axes[0])
        elif action == ActionMoRo12IntEnum.ROTATE_PITCH_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step) * axes[1])
        elif action == ActionMoRo12IntEnum.ROTATE_PITCH_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step) * axes[1])
        elif action == ActionMoRo12IntEnum.ROTATE_YAW_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step) * axes[2])
        elif action == ActionMoRo12IntEnum.ROTATE_YAW_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step) * axes[2])
        else:
            raise NotImplementedError
        return relative_move, relative_rot

    def get_action_size(self):
        return 12


class ActionMoRo15(object):

    def get_relative_move_rot2(self, axes, action, move_step, rot_step):
        relative_move = np.array([0, 0, 0])
        relative_rot = Rotation.from_quat(np.array([0, 0, 0, 1.0]))
        if action == ActionMoRo15IntEnum.MOVE_FORWARD:
            relative_move = axes[0] * move_step
        elif action == ActionMoRo15IntEnum.MOVE_BACKWARD:
            relative_move = -axes[0] * move_step
        elif action == ActionMoRo15IntEnum.MOVE_LEFT:
            relative_move = axes[1] * move_step
        elif action == ActionMoRo15IntEnum.MOVE_RIGHT:
            relative_move = -axes[1] * move_step
        elif action == ActionMoRo15IntEnum.MOVE_UP:
            relative_move = axes[2] * move_step
        elif action == ActionMoRo15IntEnum.MOVE_DOWN:
            relative_move = -axes[2] * move_step
        elif action == ActionMoRo15IntEnum.ROTATE_ROLL_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step) * axes[0])
        elif action == ActionMoRo15IntEnum.ROTATE_ROLL_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step) * axes[0])
        elif action == ActionMoRo15IntEnum.ROTATE_PITCH_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step) * axes[1])
        elif action == ActionMoRo15IntEnum.ROTATE_PITCH_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step) * axes[1])
        elif action == ActionMoRo15IntEnum.ROTATE_YAW_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step) * axes[2])
        elif action == ActionMoRo15IntEnum.ROTATE_YAW_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step) * axes[2])
        elif action == ActionMoRo15IntEnum.SMALL_STEP:
            pass
        else:
            raise NotImplementedError
        return relative_move, relative_rot

    def get_action_size(self):
        return 12


class ActionMoRoMultiplier36(object):

    def __init__(self):
        self.action_space = []
        for action in ActionMoRo12IntEnum:
            for mtplier in MultiplierIntEnum:
                self.action_space.append([action, mtplier])

    def get_relative_move_rot2(self, axes, action_ind, move_step, rot_step):
        relative_move = np.array([0, 0, 0])
        relative_rot = Rotation.from_quat(np.array([0, 0, 0, 1.0]))
        print(action_ind)
        action, multiplier = self.action_space[action_ind]

        if action == ActionMoRo12IntEnum.MOVE_FORWARD:
            relative_move = axes[0] * move_step * multiplier
        elif action == ActionMoRo12IntEnum.MOVE_BACKWARD:
            relative_move = -axes[0] * move_step * multiplier
        elif action == ActionMoRo12IntEnum.MOVE_LEFT:
            relative_move = axes[1] * move_step * multiplier
        elif action == ActionMoRo12IntEnum.MOVE_RIGHT:
            relative_move = -axes[1] * move_step * multiplier
        elif action == ActionMoRo12IntEnum.MOVE_UP:
            relative_move = axes[2] * move_step * multiplier
        elif action == ActionMoRo12IntEnum.MOVE_DOWN:
            relative_move = -axes[2] * move_step * multiplier
        elif action == ActionMoRo12IntEnum.ROTATE_ROLL_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step * multiplier) * axes[0])
        elif action == ActionMoRo12IntEnum.ROTATE_ROLL_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step * multiplier) * axes[0])
        elif action == ActionMoRo12IntEnum.ROTATE_PITCH_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step * multiplier) * axes[1])
        elif action == ActionMoRo12IntEnum.ROTATE_PITCH_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step * multiplier) * axes[1])
        elif action == ActionMoRo12IntEnum.ROTATE_YAW_P:
            relative_rot = Rotation.from_rotvec(np.radians(rot_step * multiplier) * axes[2])
        elif action == ActionMoRo12IntEnum.ROTATE_YAW_N:
            relative_rot = Rotation.from_rotvec(np.radians(-rot_step * multiplier) * axes[2])
        else:
            raise NotImplementedError
        return relative_move, relative_rot

    def get_action_size(self):
        return len(ActionMoRo12IntEnum) * len(MultiplierIntEnum)


class ActionMoRoContinuous(object):
    def get_action_size(self, robot_pos, robot_rot):
        return len(robot_pos) + len(robot_rot)


class ActionMoRo12IntEnum(IntEnum):
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


class ActionMoRo15IntEnum(IntEnum):
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
    SMALL_STEP = 12
    MEDIUM_STEP = 13
    LARGE_STEP = 14


class MultiplierIntEnum(IntEnum):
    LARGE_STEP = 3
    MEDIUM_STEP = 2
    SMALL_STEP = 1


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


# class Action2:
#     def __init__(self, a, b):
#         self.a = a
#         self.step = b
#
#     def build_actions():
#         action_space = []
#         for action in Action:

if __name__ == '__main__':
    ac = ActionMoRoMultiplier36()
