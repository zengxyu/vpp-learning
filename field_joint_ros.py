#!/usr/bin/env python

import numpy as np
from enum import IntEnum
from scipy.spatial.transform import Rotation
import binvox_rw
import time
import field_env_3d_helper
from field_env_3d_helper import Vec3D

from scripts.vpp_env_client import EnvironmentClient

vec_apply = np.vectorize(Rotation.apply, otypes=[np.ndarray], excluded=['vectors', 'inverse'])


def generate_vec3d_from_arr(arr):
    return Vec3D(*tuple(arr))


generate_vec3d_vectorized = np.vectorize(generate_vec3d_from_arr, otypes=[Vec3D])

count_unknown_vectorized = np.vectorize(field_env_3d_helper.count_unknown, otypes=[int], excluded=[0, 1, 3, 4])
count_known_free_vectorized = np.vectorize(field_env_3d_helper.count_known_free, otypes=[int], excluded=[0, 1, 3, 4])
count_known_target_vectorized = np.vectorize(field_env_3d_helper.count_known_target, otypes=[int],
                                             excluded=[0, 1, 3, 4])

count_unknown_layer5_vectorized = np.vectorize(field_env_3d_helper.count_unknown_layer5,
                                               otypes=[int, int, int, int, int], excluded=[0, 1, 3, 4])
count_known_free_layer5_vectorized = np.vectorize(field_env_3d_helper.count_known_free_layer5,
                                                  otypes=[int, int, int, int, int], excluded=[0, 1, 3, 4])
count_known_target_layer5_vectorized = np.vectorize(field_env_3d_helper.count_known_target_layer5,
                                                    otypes=[int, int, int, int, int], excluded=[0, 1, 3, 4])


#
# class Action(IntEnum):
#     DO_NOTHING = 0
#     MOVE_FORWARD = 1
#     MOVE_BACKWARD = 2
#     MOVE_LEFT = 3
#     MOVE_RIGHT = 4
#     MOVE_UP = 5
#     MOVE_DOWN = 6
#     ROTATE_ROLL_P = 7
#     ROTATE_ROLL_N = 8
#     ROTATE_PITCH_P = 9
#     ROTATE_PITCH_N = 10
#     ROTATE_YAW_P = 11
#     ROTATE_YAW_N = 12

class Action(IntEnum):
    MOVE_JOINT0_FORWARD = 0
    MOVE_JOINT0_BACKWARD = 1
    MOVE_JOINT1_FORWARD = 2
    MOVE_JOINT1_BACKWARD = 3
    MOVE_JOINT2_FORWARD = 4
    MOVE_JOINT2_BACKWARD = 5
    MOVE_JOINT3_FORWARD = 6
    MOVE_JOINT3_BACKWARD = 7
    MOVE_JOINT4_FORWARD = 8
    MOVE_JOINT4_BACKWARD = 9


# class Action(IntEnum):
#     MOVE_FORWARD = 0,
#     ROTATE_ROLL_P = 1,
#     ROTATE_ROLL_N = 2,
#     ROTATE_PITCH_P = 3,
#     ROTATE_PITCH_N = 4,
#     ROTATE_YAW_P = 5,
#     ROTATE_YAW_N = 6

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
        if init_file:
            self.read_env_from_file(init_file, scale)

    def read_env_from_file(self, filename, scale):
        with open(filename, 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
        self.global_map = np.transpose(model.data, (2, 0, 1)).astype(int)
        self.target_count = np.count_nonzero(self.global_map)
        print("Total target count : {} ".format(self.target_count))
        print("#targets/#free_cells = {}".format(self.target_count / (np.product(self.shape))))
        self.found_targets = 0
        self.free_cells = 0
        self.global_map += 1  # Shift: 1 - free, 2 - occupied/target
        self.shape = self.global_map.shape
        self.known_map = np.zeros(self.shape)

        if not self.headless:
            from field_env_3d_gui import FieldGUI
            self.gui = FieldGUI(self, scale)

    def compute_fov(self):
        axes = self.robot_rot.as_matrix().transpose()
        rh = np.radians(self.hfov / 2)
        rv = np.radians(self.vfov / 2)
        vec_left_down = (Rotation.from_rotvec(rh * axes[2]) * Rotation.from_rotvec(rv * axes[1])).apply(axes[0])
        vec_left_up = (Rotation.from_rotvec(rh * axes[2]) * Rotation.from_rotvec(-rv * axes[1])).apply(axes[0])
        vec_right_down = (Rotation.from_rotvec(-rh * axes[2]) * Rotation.from_rotvec(rv * axes[1])).apply(axes[0])
        vec_right_up = (Rotation.from_rotvec(-rh * axes[2]) * Rotation.from_rotvec(-rv * axes[1])).apply(axes[0])
        ep_left_down = self.robot_pos + vec_left_down * self.sensor_range
        ep_left_up = self.robot_pos + vec_left_up * self.sensor_range
        ep_right_down = self.robot_pos + vec_right_down * self.sensor_range
        ep_right_up = self.robot_pos + vec_right_up * self.sensor_range
        return self.robot_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up

    def compute_rot_vecs(self, min_ang, max_ang, num_steps):
        axes = self.robot_rot.as_matrix().transpose()
        rh = np.radians(np.linspace(min_ang, max_ang, num_steps))
        rv = np.radians(np.linspace(min_ang, max_ang, num_steps))
        rots_x = Rotation.from_rotvec(np.outer(rh, axes[2]))
        rots_y = Rotation.from_rotvec(np.outer(rv, axes[1]))
        rots = vec_apply(np.outer(rots_x, rots_y), vectors=axes[0])
        rot_vecs = generate_vec3d_vectorized(rots)
        return rot_vecs

    def generate_unknown_map(self, cam_pos):
        rot_vecs = self.compute_rot_vecs(-180, 180, 18)
        # unknown_map = count_unknown_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos), rot_vecs, 1.0, 50.0)
        # known_free_map = count_known_free_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos), rot_vecs, 1.0,
        #                                              50.0)
        # known_target_map = count_known_target_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos), rot_vecs,
        #                                                  1.0, 50.0)

        unknown_map = count_unknown_layer5_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos), rot_vecs, 1.0,
                                                      250.0)
        known_free_map = count_known_free_layer5_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos), rot_vecs,
                                                            1.0, 250.0)
        known_target_map = count_known_target_layer5_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos),
                                                                rot_vecs, 1.0, 250.0)

        return unknown_map, known_free_map, known_target_map

    def line_plane_intersection(self, p0, nv, l0, lv):
        """ return intersection of a line with a plane

        Parameters:
            p0: Point in plane
            nv: Normal vector of plane
            l0: Point on line
            lv: Direction vector of line

        Returns:
            The intersection point
        """
        denom = np.dot(lv, nv)
        if denom == 0:  # No intersection or line contained in plane
            return None, None

        d = np.dot((p0 - l0), nv) / denom
        return l0 + lv * d, d

    def point_in_rectangle(self, p, p0, v1, v2):
        """ check if point is within reactangle

        Parameters:
            p: Point
            p0: Corner point of rectangle
            v1: Side vector 1 starting from p0
            v2: Side vector 2 starting from p0

        Returns:
            True if within rectangle
        """
        v1_len = np.linalg.norm(v1)
        v2_len = np.linalg.norm(v2)
        v1_proj_length = np.dot((p - p0), v1 / v1_len)
        v2_proj_length = np.dot((p - p0), v2 / v2_len)
        return (v1_proj_length >= 0 and v1_proj_length <= v1_len and v2_proj_length >= 0 and v2_proj_length <= v2_len)

    def get_bb_points(self, points):
        return np.amin(points, axis=0), np.amax(points, axis=0)

    def update_grid_inds_in_view(self, cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up):
        time_start = time.perf_counter()
        self.known_map, found_targets, free_cells, coords, values = field_env_3d_helper.update_grid_inds_in_view(
            self.known_map,
            self.global_map,
            Vec3D(*tuple(
                cam_pos)),
            Vec3D(*tuple(
                ep_left_down)),
            Vec3D(*tuple(
                ep_left_up)),
            Vec3D(*tuple(
                ep_right_down)),
            Vec3D(*tuple(
                ep_right_up)))

        if not self.headless:
            self.gui.messenger.send('update_fov_and_cells',
                                    [cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up,
                                     coords, values], 'default')
            self.gui.gui_done.wait()
            self.gui.gui_done.clear()

        # print("Updating field took {} s".format(time.perf_counter() - time_start))

        return found_targets, free_cells

    def update_grid_inds_in_view_old(self, cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up):
        time_start = time.perf_counter()
        bb_min, bb_max = self.get_bb_points([cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up])
        bb_min, bb_max = np.clip(np.rint(bb_min), [0, 0, 0], self.shape).astype(int), np.clip(np.rint(bb_max),
                                                                                              [0, 0, 0],
                                                                                              self.shape).astype(int)
        v1 = ep_right_up - ep_right_down
        v2 = ep_left_down - ep_right_down
        plane_normal = np.cross(v1, v2)
        found_targets = 0
        if not self.headless:
            coords = PTA_int()
            values = PTA_int()
        for z in range(bb_min[2], bb_max[2]):
            for y in range(bb_min[1], bb_max[1]):
                for x in range(bb_min[0], bb_max[0]):
                    point = np.array([x, y, z])
                    if self.known_map[x, y, z] != FieldValues.UNKNOWN:  # no update necessary if point already seen
                        continue
                    p_proj, rel_dist = self.line_plane_intersection(ep_right_down, plane_normal, cam_pos,
                                                                    (point - cam_pos))
                    if p_proj is None or rel_dist < 1.0:  # if point lies behind projection, skip
                        continue
                    if self.point_in_rectangle(p_proj, ep_right_down, v1, v2):
                        self.known_map[x, y, z] = self.global_map[x, y, z]
                        # for now, occupied cells are targets, change later
                        if self.known_map[x, y, z] == FieldValues.OCCUPIED:
                            found_targets += 1
                        if not self.headless:
                            coords.push_back(int(x))
                            coords.push_back(int(y))
                            coords.push_back(int(z))
                            values.push_back(int(self.known_map[x, y, z] + 3))
                            # self.gui.messenger.send('update_cell', [(x, y, z)], 'default')
                            # self.gui.gui_done.wait()
                            # self.gui.gui_done.clear()
                            # self.gui.updateSeenCell((x, y, z))

        if not self.headless:
            self.gui.messenger.send('update_fov_and_cells',
                                    [cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up, coords, values],
                                    'default')
            # self.gui.messenger.send('update_fov', [cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up], 'default')
            self.gui.gui_done.wait()
            self.gui.gui_done.clear()
            # self.gui.updateFov(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)

        print("Updating field took {} s".format(time.perf_counter() - time_start))

        return found_targets

    def move_robot(self, direction):
        self.robot_pos += direction
        # self.robot_pos = np.clip(self.robot_pos, self.allowed_lower_bound, self.allowed_upper_bound)

    def rotate_robot(self, axis, angle):
        rot = Rotation.from_rotvec(np.radians(angle) * axis)
        self.robot_rot = rot * self.robot_rot

    def relative_rotation(self, axis, angle):
        rot = Rotation.from_rotvec(np.radians(angle) * axis)
        return rot.as_quat()

    # def step(self, action):
    #     axes = self.robot_rot.as_matrix().transpose()
    #     relative_move = np.array([0, 0, 0])
    #     relative_rot = np.array([0, 0, 0, 0])
    #     if action == Action.MOVE_FORWARD:
    #         relative_move = np.array([0.1, 0, 0])
    #     elif action == Action.MOVE_BACKWARD:
    #         relative_move = np.array([-0.1, 0, 0])
    #     elif action == Action.MOVE_LEFT:
    #         relative_move = np.array([0, 0.1, 0])
    #     elif action == Action.MOVE_RIGHT:
    #         relative_move = np.array([0, -0.1, 0])
    #     elif action == Action.MOVE_UP:
    #         relative_move = np.array([0, 0, 0.1])
    #     elif action == Action.MOVE_DOWN:
    #         relative_move = np.array([0, 0, -0.1])
    #     elif action == Action.ROTATE_ROLL_P:
    #         r = Rotation.from_euler('x', self.ROT_STEP, degrees=True)
    #         relative_rot = r.as_quat()
    #     elif action == Action.ROTATE_ROLL_N:
    #         r = Rotation.from_euler('x', -self.ROT_STEP, degrees=True)
    #         relative_rot = r.as_quat()
    #     elif action == Action.ROTATE_PITCH_P:
    #         relative_rot = self.relative_rotation(axes[1], self.ROT_STEP)
    #         self.rotate_robot(axes[1], self.ROT_STEP)
    #         r = Rotation.from_euler('y', self.ROT_STEP, degrees=True)
    #         relative_rot = r.as_quat()
    #     elif action == Action.ROTATE_PITCH_N:
    #         r = Rotation.from_euler('y', -self.ROT_STEP, degrees=True)
    #         relative_rot = r.as_quat()
    #     elif action == Action.ROTATE_YAW_N:
    #         relative_rot = self.relative_rotation(axes[2], self.ROT_STEP)
    #         self.rotate_robot(axes[2], self.ROT_STEP)
    #         r = Rotation.from_euler('z', -self.ROT_STEP, degrees=True)
    #         relative_rot = r.as_quat()
    #     elif action == Action.ROTATE_YAW_P:
    #         r = Rotation.from_euler('z', self.ROT_STEP, degrees=True)
    #         relative_rot = r.as_quat()
    #     relative_pose = np.append(relative_move, relative_rot).tolist()
    #     start_time = time.time()
    #     unknownCount, freeCount, occupiedCount, roiCount, robotPose, robotJoints, reward = self.client.sendRelativePose(
    #         relative_pose)
    #     # print("robot pose computed by me:{}".format(self.robot_pos))
    #     self.robot_pos = robotPose[:3]
    #     # print("robot pose computed by remote:{}".format(self.robot_pos))
    #     self.robot_rot = Rotation.from_quat(robotPose[3:])
    #     print("sendRelativeTime:{}".format(time.time() - start_time))
    #     # cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up = self.compute_fov()
    #     # new_targets_found, new_free_cells = self.update_grid_inds_in_view(cam_pos, ep_left_down, ep_left_up,
    #     #                                                                   ep_right_down, ep_right_up)
    #     # self.free_cells += new_free_cells
    #     self.found_targets += reward
    #     self.step_count += 1
    #     done = (self.found_targets == self.target_count) or (self.step_count >= self.max_steps)
    #
    #     # unknown_map, known_free_map, known_target_map = self.generate_unknown_map(cam_pos)
    #     map = np.concatenate([unknownCount, freeCount, roiCount], axis=0)
    #
    #     # return map, np.concatenate(
    #     #     (self.robot_pos, self.robot_rot.as_quat())), reward, 0, done
    #     return map, robotPose, reward, 0, done

    def step(self, action):
        joint_relative_move = np.array([0, 0, 0, 0, 0])
        if action == Action.MOVE_JOINT0_FORWARD:
            joint_relative_move = np.array([1, 0, 0, 0, 0])
        elif action == Action.MOVE_JOINT0_BACKWARD:
            joint_relative_move = np.array([-1, 0, 0, 0, 0])
        elif action == Action.MOVE_JOINT1_FORWARD:
            joint_relative_move = np.array([0, 1, 0, 0, 0])
        elif action == Action.MOVE_JOINT1_BACKWARD:
            joint_relative_move = np.array([0, -1, 0, 0, 0])
        elif action == Action.MOVE_JOINT2_FORWARD:
            joint_relative_move = np.array([0, 0, 1, 0, 0])
        elif action == Action.MOVE_JOINT2_BACKWARD:
            joint_relative_move = np.array([0, 0, -1, 0, 0])
        elif action == Action.MOVE_JOINT3_FORWARD:
            joint_relative_move = np.array([0, 0, 0, 1, 0])
        elif action == Action.MOVE_JOINT3_BACKWARD:
            joint_relative_move = np.array([0, 0, 0, -1, 0])
        elif action == Action.MOVE_JOINT4_FORWARD:
            joint_relative_move = np.array([0, 0, 0, 0, 1])
        elif action == Action.MOVE_JOINT4_BACKWARD:
            joint_relative_move = np.array([0, 0, 0, 0, -1])
        relative_move = np.deg2rad(joint_relative_move * self.ROT_STEP).tolist()
        start_time = time.time()
        unknownCount, freeCount, occupiedCount, roiCount, robotPose, robotJoints, reward = self.client.sendRelativeJointTarget(
            relative_move)
        # print("robot pose computed by me:{}".format(self.robot_pos))
        # self.robot_pos = robotPose[:3]
        # print("robot pose computed7
        # by remote:{}".format(self.robot_pos))
        # self.robot_rot = Rotation.from_quat(robotPose[3:])
        print("sendRelativeTime:{}".format(time.time() - start_time))
        # cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up = self.compute_fov()
        # new_targets_found, new_free_cells = self.update_grid_inds_in_view(cam_pos, ep_left_down, ep_left_up,
        #                                                                   ep_right_down, ep_right_up)
        # self.free_cells += new_free_cells
        self.found_targets += reward
        self.step_count += 1
        done = (self.found_targets == self.target_count) or (self.step_count >= self.max_steps)

        # unknown_map, known_free_map, known_target_map = self.generate_unknown_map(cam_pos)
        map = np.concatenate([unknownCount, freeCount, roiCount], axis=0)

        # return map, np.concatenate(
        #     (self.robot_pos, self.robot_rot.as_quat())), reward, 0, done
        return map, robotPose, robotJoints, reward, done

    def reset(self):
        self.reset_count += 1
        self.known_map = np.zeros(self.shape)
        # self.observed_area = np.zeros(self.shape, dtype=bool)
        # self.allowed_range = np.array([128, 128, 128])
        # self.allowed_lower_bound = np.array([128, 128, 128]) - self.allowed_range
        # self.allowed_upper_bound = np.array([128, 128, 128]) + self.allowed_range - 1
        # self.allowed_lower_bound = np.array([0, 0, 0])
        # self.allowed_lower_bound = np.array([128, 128, 128])

        # if self.reset_count % 2 == 0:
        #     self.upper_scale += 1
        # upper = np.array([1.0, 1.0, 1.0]) * self.upper_scale
        #
        # # self.robot_pos = np.random.uniform(self.allowed_lower_bound, self.allowed_upper_bound)
        # upper = np.clip(upper, np.array([0.0, 0.0, 0.0]), np.array([255.0, 255.0, 255.0]))
        # self.robot_pos = np.random.uniform(np.array([0.0, 0.0, 0.0]), np.array([255.0, 255.0, 255.0]))
        # # self.robot_pos = np.array([0.0, 0.0, 0.0])
        #
        # # print("upper:{}; reset robot pose as:{}".format(upper, self.robot_pos))
        # print("reset robot pose as:{}".format(self.robot_pos))

        # self.robot_pos = np.array([0.0, 0.0, 0.0])

        # self.robot_rot = Rotation.random()
        # self.robot_rot = Rotation.from_quat([0, 0, 0, 1])
        # print("allowed range:{}".format(self.allowed_range))
        self.step_count = 0
        self.found_targets = 0
        self.free_cells = 0

        if not self.headless:
            self.gui.messenger.send('reset', [], 'default')
            self.gui.gui_done.wait()
            self.gui.gui_done.clear()
            # self.gui.reset()
        # relative_pose = np.array([0, 0, 0, 0, 0, 0, 0]).tolist()
        # cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up = self.compute_fov()
        # self.update_grid_inds_in_view(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)
        unknownCount, freeCount, occupiedCount, roiCount, robotPose, robotJoints, reward = self.client.sendReset()
        # print(self.robot_pos)
        # print(self.robot_rot.as_quat())
        # unknown_map, known_free_map, known_target_map = self.generate_unknown_map(cam_pos)
        # print(unknown_map)
        map = np.concatenate([unknownCount, freeCount, roiCount], axis=0)

        # return map, np.concatenate((self.robot_pos, self.robot_rot.as_quat()))
        return map, robotPose, robotJoints
