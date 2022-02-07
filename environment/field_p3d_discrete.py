#!/usr/bin/environment python
import os.path

import numpy as np
from enum import IntEnum
from scipy.spatial.transform import Rotation
import binvox_rw
import time
import field_env_3d_helper
from field_env_3d_helper import Vec3D

from config import read_yaml, get_configs_dir
from environment.utilities.map_helper import make_up_map, make_up_8x15x9x9_map
from environment.utilities.random_field_helper import random_translate_environment, expand_and_randomize_environment
from utilities.util import get_project_path

import capnp
import voxelgrid_capnp

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

count_unknown_layer2_vectorized = np.vectorize(field_env_3d_helper.count_unknown_layer2,
                                               otypes=[int, int], excluded=[0, 1, 3, 4])
count_known_free_layer2_vectorized = np.vectorize(field_env_3d_helper.count_known_free_layer2,
                                                  otypes=[int, int], excluded=[0, 1, 3, 4])
count_known_target_layer2_vectorized = np.vectorize(field_env_3d_helper.count_known_target_layer2,
                                                    otypes=[int, int], excluded=[0, 1, 3, 4])


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
    def __init__(self, config, action_space):
        self.config = config
        env_config = read_yaml(config_dir=get_configs_dir(), config_name="env.yaml")
        self.env_config = env_config
        self.sensor_range = env_config["sensor_range"]
        self.hfov = env_config["hfov"]
        self.vfov = env_config["vfov"]
        self.shape = (env_config["shape_z"], env_config["shape_x"], env_config["shape_y"])
        self.max_steps = env_config["max_steps"]
        self.headless = env_config["headless"]

        self.action_space = action_space
        self.global_map = np.zeros(self.shape).astype(int)
        self.known_map = np.zeros(self.shape).astype(int)
        # how often to augment the environment
        self.robot_pos = [0.0, 0.0, 0.0]
        self.robot_rot = Rotation.from_quat([0, 0, 0, 1])
        self.MOVE_STEP = env_config["move_step"]
        self.ROT_STEP = env_config["rot_step"]
        self.randomize = env_config["randomize"]

        self.reset_count = 0
        self.target_count = 0
        self.free_count = 0
        self.found_targets = 0
        self.free_cells = 0

        self.allowed_range = np.array([128, 128, 128])
        self.allowed_lower_bound = np.array([128, 128, 128]) - self.allowed_range
        self.allowed_upper_bound = np.array([128, 128, 128]) + self.allowed_range - 1

        self.visit_resolution = 8
        self.visit_shape = None
        self.visit_map = None

        self.init_file_path = os.path.join(get_project_path(), 'saved_world.cvx')
        # self.read_env_from_file(self.init_file_path)
        self.read_env_from_capnp(self.init_file_path)

        print("max steps:", self.max_steps)
        print("move step:", self.MOVE_STEP)
        print("rot step:", self.ROT_STEP)
        if not self.headless:
            from environment.field_p3d_gui import FieldGUI
            self.gui = FieldGUI(self, env_config["scale"])

    def read_env_from_capnp(self, filename):
        with open(filename) as file:
            voxelgrid = voxelgrid_capnp.Voxelgrid.read(file, traversal_limit_in_words=2**32)
        
        labels = np.asarray(voxelgrid.labels) + 1
        self.shape = tuple(voxelgrid.shape)
        self.global_map = labels.reshape(self.shape)
        
        self.visit_shape = (int(self.shape[0] // self.visit_resolution), int(self.shape[1] // self.visit_resolution),
                            int(self.shape[2] // self.visit_resolution))
        self.visit_map = np.zeros(shape=self.visit_shape, dtype=np.uint8)

        self.found_targets = 0
        self.free_cells = 0

        self.target_count = np.sum(self.global_map == 2)
        self.free_count = np.sum(self.global_map == 1)

        print("#targets = {}".format(self.target_count))
        print("#free = {}".format(self.free_count))

        print("#targets/#total = {}".format(self.target_count / (np.product(self.shape))))
        print("#free/#total = {}".format(self.free_count / (np.product(self.shape))))
        print("total reward = {}".format(self.target_count + 0.01 * self.free_count))

    def read_env_from_file(self, filename):
        self.global_map = np.zeros(self.shape).astype(int)
        self.known_map = np.zeros(self.shape).astype(int)

        with open(filename, 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
        read_model = np.transpose(model.data, (2, 0, 1)).astype(int)
        rs = read_model.shape
        self.global_map[0:rs[0], 0:rs[1], 0:rs[2]] = read_model
        # self.global_map = np.transpose(model.data, (2, 0, 1)).astype(int)
        # self.global_map = expand_and_randomize_environment(self.global_map, self.shape)

        self.global_map += 1  # Shift: 1 - free, 2 - occupied/target
        self.shape = self.global_map.shape

        self.visit_shape = (int(self.shape[0] // self.visit_resolution), int(self.shape[1] // self.visit_resolution),
                            int(self.shape[2] // self.visit_resolution))
        self.visit_map = np.zeros(shape=self.visit_shape, dtype=np.uint8)

        self.found_targets = 0
        self.free_cells = 0

        self.target_count = np.sum(self.global_map == 2)
        self.free_count = np.sum(self.global_map == 1)

        print("#targets = {}".format(self.target_count))
        print("#free = {}".format(self.free_count))

        print("#targets/#total = {}".format(self.target_count / (np.product(self.shape))))
        print("#free/#total = {}".format(self.free_count / (np.product(self.shape))))
        print("total reward = {}".format(self.target_count + 0.01 * self.free_count))

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

    def compute_rot_vecs(self, min_ang_width, max_ang_width, width_steps, min_ang_height, max_ang_height, height_steps):
        axes = self.robot_rot.as_matrix().transpose()
        rh = np.radians(np.linspace(min_ang_width, max_ang_width, width_steps))
        rv = np.radians(np.linspace(min_ang_height, max_ang_height, height_steps))
        rots_x = Rotation.from_rotvec(np.outer(rh, axes[2]))
        rots_y = Rotation.from_rotvec(np.outer(rv, axes[1]))
        rots = vec_apply(np.outer(rots_x, rots_y), vectors=axes[0])
        rot_vecs = generate_vec3d_vectorized(rots)
        return rot_vecs

    def generate_unknown_map_layer5(self, cam_pos, dist=250.0):
        rot_vecs = self.compute_rot_vecs(-180, 180, 36, 0, 180, 18)

        unknown_map = count_unknown_layer5_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos), rot_vecs,
                                                      1.0, dist)
        known_free_map = count_known_free_layer5_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos),
                                                            rot_vecs, 1.0, dist)
        known_target_map = count_known_target_layer5_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos),
                                                                rot_vecs, 1.0, dist)
        return unknown_map, known_free_map, known_target_map

    def generate_unknown_map_layer2(self, cam_pos, dist=250.0):
        rot_vecs = self.compute_rot_vecs(-180, 180, 36, 0, 180, 18)

        unknown_map = count_unknown_layer2_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos), rot_vecs,
                                                      1.0, dist)
        known_free_map = count_known_free_layer2_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos),
                                                            rot_vecs, 1.0, dist)
        known_target_map = count_known_target_layer2_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos),
                                                                rot_vecs, 1.0, dist)
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

        # target_count = np.sum(self.known_map == 2)
        # free_count = np.sum(self.known_map == 1)
        # print("\n#targets/#total = {}".format(target_count / (np.product(self.shape))))
        # print("#free/#total = {}".format(free_count / (np.product(self.shape))))

        # print("Updating field took {} s".format(time.perf_counter() - time_start))
        return found_targets, free_cells

    def move_robot(self, direction):
        self.robot_pos += direction
        self.robot_pos = np.clip(self.robot_pos, self.allowed_lower_bound, self.allowed_upper_bound)

    def rotate_robot(self, rot):
        self.robot_rot = rot * self.robot_rot

    def concat(self, unknown_map, known_free_map, known_target_map):
        map = np.concatenate([unknown_map, known_free_map, known_target_map], axis=0)
        return map

    def generate_spherical_coordinate_map(self, cam_pos):
        rot_vecs = self.compute_rot_vecs(-180, 180, 36, 0, 180, 18)
        spherical_coordinate_map = field_env_3d_helper.generate_spherical_coordinate_map(self.known_map,
                                                                                         generate_vec3d_from_arr(
                                                                                             cam_pos), rot_vecs, 250.0,
                                                                                         250)
        spherical_coordinate_map = np.transpose(spherical_coordinate_map, (2, 0, 1))
        return spherical_coordinate_map

    def compute_global_known_map(self, cam_pos, neighbor_dist):
        generate_spherical_coordinate_map = self.generate_spherical_coordinate_map(cam_pos)
        step_size = 10
        res = np.zeros(shape=(2, int(neighbor_dist / step_size), 36, 18))

        for i in range(0, neighbor_dist, step_size):
            res[0, i // step_size, :, :] = np.sum(generate_spherical_coordinate_map[:, :, i:i + step_size] == 1)
            res[1, i // step_size, :, :] = np.sum(generate_spherical_coordinate_map[:, :, i:i + step_size] == 2)

        res = np.concatenate((res[0], res[1]), axis=0)
        return res

    def step(self, action):
        axes = self.robot_rot.as_matrix().transpose()
        relative_move, relative_rot = self.action_space.get_relative_move_rot(axes, action, self.MOVE_STEP,
                                                                              self.ROT_STEP)
        self.move_robot(relative_move)
        self.rotate_robot(relative_rot)
        cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up = self.compute_fov()
        new_found_targets, new_free_cells = self.update_grid_inds_in_view(cam_pos,
                                                                          ep_left_down,
                                                                          ep_left_up,
                                                                          ep_right_down,
                                                                          ep_right_up)
        visit_gain, coverage_rate = self.update_visit_map()

        self.found_targets += new_found_targets
        self.free_cells += new_free_cells

        self.step_count += 1
        done = (self.found_targets == self.target_count) or (self.step_count >= self.max_steps)
        # 5 * 36 * 18
        unknown_map, known_free_map, known_target_map = self.generate_unknown_map_layer5(cam_pos)

        # 15 * 36 * 18
        map = self.concat(unknown_map, known_free_map, known_target_map)

        # map = make_up_8x15x9x9_map(map)

        map = map.astype(np.uint8)
        reward = 5000 * visit_gain + new_found_targets + 0.01 * new_free_cells
        # reward = 200 * new_found_targets + new_free_cells

        # step
        # print("new_found_targets:{}".format(new_found_targets))
        # print("new_free_cells:{}".format(new_free_cells))
        info = {"visit_gain": visit_gain, "new_found_targets": new_found_targets, "new_free_cells": new_free_cells,
                "reward": reward, "coverage_rate": coverage_rate}
        # print(info)
        return map, reward, done, info

    def update_visit_map(self):
        # to encourage exploration, 计算整个图新找到的target的数量， 动作变成36, 鼓励去没去过的地方
        # 去大的没有去过的格子
        # need LSTM
        location = np.array([self.robot_pos[0] // self.visit_resolution, self.robot_pos[1] // self.visit_resolution,
                             self.robot_pos[2] // self.visit_resolution]).astype(np.int)
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

        self.visit_map[location[0], location[1], location[2]] = 1
        coverage_rate = np.sum(self.visit_map) / np.product(self.visit_shape)

        return visit_gain, coverage_rate

    def reset(self):
        self.reset_count += 1
        if self.randomize:
            self.read_env_from_capnp(self.init_file_path)

        self.robot_pos = np.array([0.0, 0.0, 0.0])
        self.robot_rot = Rotation.from_quat([0, 0, 0, 1])

        self.step_count = 0
        self.found_targets = 0
        self.free_cells = 0

        if not self.headless:
            self.gui.messenger.send('reset', [], 'default')
            self.gui.gui_done.wait()
            self.gui.gui_done.clear()
            # self.gui.reset()

        cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up = self.compute_fov()
        self.update_grid_inds_in_view(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)

        unknown_map, known_free_map, known_target_map = self.generate_unknown_map_layer5(cam_pos)
        # known_target_rate, unknown_rate = self.count_neighbor_rate(np.array(unknown_map), np.array(known_free_map),
        #                                                            np.array(known_target_map))
        map = self.concat(unknown_map, known_free_map, known_target_map)

        # map = make_up_8x15x9x9_map(map)
        map = map.astype(np.uint8)

        return map, {}
