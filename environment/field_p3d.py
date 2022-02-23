#!/usr/bin/environment python
import os.path
import pickle
import sys
import time

import field_env_3d_helper
from field_env_3d_helper import Vec3D
from scipy.spatial.transform import Rotation

from environment.utilities.bound_helper import get_sensor_position_bound, get_world_bound, in_plant_bounds, \
    out_of_world_bound, out_of_sensor_position_bound, initialize_world_shape
from environment.utilities.count_cells_helper import count_observable_cells, count_cells
from environment.utilities.map_concat_helper import concat
from environment.utilities.plant_models_loader import load_plants
from environment.utilities.random_plant_position_helper import get_random_multi_plant_models, \
    get_random_number_of_plants
from environment.utilities.randomize_camera_position import randomize_camera_position
from environment.utilities.save_observation_map_helper import save_observation_map

from utilities.util import get_project_path
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "capnp"))
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
count_known_occupied_layer5_vectorized = np.vectorize(field_env_3d_helper.count_known_occupied_layer5,
                                                      otypes=[int, int, int, int, int], excluded=[0, 1, 3, 4])
count_known_target_layer5_vectorized = np.vectorize(field_env_3d_helper.count_known_target_layer5,
                                                    otypes=[int, int, int, int, int], excluded=[0, 1, 3, 4])

count_unknown_layer8_vectorized = np.vectorize(field_env_3d_helper.count_unknown_layer8,
                                               otypes=[int, int, int, int, int, int, int, int], excluded=[0, 1, 3, 4])
count_known_free_layer8_vectorized = np.vectorize(field_env_3d_helper.count_known_free_layer8,
                                                  otypes=[int, int, int, int, int, int, int, int],
                                                  excluded=[0, 1, 3, 4])
count_known_occupied_layer8_vectorized = np.vectorize(field_env_3d_helper.count_known_occupied_layer8,
                                                      otypes=[int, int, int, int, int, int, int, int],
                                                      excluded=[0, 1, 3, 4])
count_known_target_layer8_vectorized = np.vectorize(field_env_3d_helper.count_known_target_layer8,
                                                    otypes=[int, int, int, int, int, int, int, int],
                                                    excluded=[0, 1, 3, 4])


class FieldP3D:
    def __init__(self, parser_args, action_space):
        self.parser_args = parser_args
        self.env_config = parser_args.env_config
        self.training_config = parser_args.training_config
        self.max_sensor_range = self.env_config["sensor_range"][1]

        self.hfov = self.env_config["hfov"]
        self.vfov = self.env_config["vfov"]
        self.hrays = self.env_config["hrays"]
        self.vrays = self.env_config["vrays"]

        self.max_steps = self.env_config["max_steps"]
        self.MOVE_STEP = self.env_config["move_step"]
        self.ROT_STEP = self.env_config["rot_step"]

        self.obs_hrange = self.env_config["obs_hrange"]
        self.obs_vrange = self.env_config["obs_vrange"]
        self.obs_drange = self.env_config["obs_drange"]
        self.obs_hsteps = self.env_config["obs_hsteps"]
        self.obs_vsteps = self.env_config["obs_vsteps"]

        self.head = parser_args.head
        self.randomize_plant_position = self.env_config["randomize_plant_position"]
        self.randomize_sensor_position = self.env_config["randomize_sensor_position"]
        self.randomize_world_size = self.env_config["randomize_world_size"]
        self.random_plant_number = self.env_config["random_plant_number"]

        self.thresh = self.env_config["thresh"]
        self.plant_position_margin = self.env_config["plant_position_margin"]
        self.sensor_position_margin = self.env_config["sensor_position_margin"]
        self.action_space = action_space
        self.reset_count = 0

        # following variables need to be reset
        self.shape = None
        self.global_map = None
        self.known_map = None

        self.robot_pos = np.array([0.0, 0.0, 0.0])
        self.robot_rot = Rotation.from_quat([0, 0, 0, 1])
        self.relative_position = np.array([0., 0., 0.])
        self.relative_rotation = Rotation.from_quat([0, 0, 0, 1])

        self.roi_total = 0
        self.occ_total = 0
        self.free_total = 0
        self.observable_roi_total = 0
        self.observable_occ_total = 0
        self.observable_free_total = 0

        self.found_roi_sum = 0
        self.found_occ_sum = 0
        self.found_free_sum = 0

        self.step_count = 0

        self.world_bound = None
        self.sensor_position_bound = None

        self.collision_count = 0
        self.stuck_count = 0

        self.visit_resolution = 16
        self.visit_shape = None
        self.visit_map = None
        self.map = None
        self.plant_bounding_boxes = None
        self.plant_types = None
        self.plants = None

        self.path_coords = []

        self.plant_models_dir = os.path.join(get_project_path(), "data", 'plant_models')

        self.all_plant_types = self.env_config["plant_types"]
        self.all_plants = load_plants(self.plant_models_dir, self.env_config["plant_types"],
                                      self.env_config["roi_neighbors"], self.env_config["resolution"])

        self.initialize()

        print("max steps:", self.max_steps)
        print("move step:", self.MOVE_STEP)
        print("rot step:", self.ROT_STEP)
        if self.head:
            from environment.utilities.field_p3d_gui import FieldGUI
            self.gui = FieldGUI(self, self.env_config["scale"])

    def initialize(self):
        self.shape = initialize_world_shape(self.env_config, self.randomize_world_size)

        self.global_map = np.zeros(self.shape).astype(int)
        self.known_map = np.zeros(self.shape).astype(int)

        self.robot_pos = np.array([0, 0, 0])
        self.robot_rot = Rotation.from_quat([0, 0, 0, 1])

        self.relative_position = np.array([0., 0., 0.])
        self.relative_rotation = Rotation.from_quat([0, 0, 0, 1])

        self.step_count = 0
        self.found_roi_sum = 0
        self.found_occ_sum = 0
        self.found_free_sum = 0

        self.world_bound = get_world_bound(self.shape)
        self.sensor_position_bound = get_sensor_position_bound(self.world_bound, self.sensor_position_margin)

        self.plant_types = self.all_plant_types
        self.plants = self.all_plants
        if self.random_plant_number:
            self.plant_types, self.plants = get_random_number_of_plants(self.env_config["plant_num_choices"],
                                                                        self.all_plant_types, self.all_plants)
        print("Plants number : {} ".format(len(self.plants)))
        print("Plants types : {} ".format(self.plant_types))

        # insert the plants randomly into the ground
        if self.randomize_plant_position:
            self.global_map, self.plant_bounding_boxes = get_random_multi_plant_models(self.global_map, self.plants,
                                                                                       self.thresh,
                                                                                       self.plant_position_margin)
        # randomize camera position if self.randomize_sensor_position True
        if self.randomize_sensor_position:
            self.robot_pos = randomize_camera_position(self.sensor_position_bound, self.plant_bounding_boxes)

        print("Sensor start position : {}", self.robot_pos)

        self.global_map += 1  # Shift: 1 - free, 2 - occupied/target
        self.roi_total, self.occ_total, self.free_total = count_cells(self.global_map)
        self.observable_roi_total, self.observable_occ_total = count_observable_cells(self.env_config, self.plant_types,
                                                                                      self.plants)

        self.collision_count = 0
        self.stuck_count = 0
        self.visit_shape = (int(self.shape[0] // self.visit_resolution),
                            int(self.shape[1] // self.visit_resolution),
                            int(self.shape[2] // self.visit_resolution))
        self.visit_map = np.zeros(shape=self.visit_shape, dtype=np.uint8)

        self.path_coords = [self.robot_pos]

        print(
            "#observable_roi = {}; #roi = {}; #observable_roi/#roi = {}".format(self.observable_roi_total,
                                                                                self.roi_total,
                                                                                self.observable_roi_total / self.roi_total))
        print("##observable_occ = {}; #occ = {}; #observable_occ/#occ = {}".format(self.observable_occ_total,
                                                                                   self.occ_total,
                                                                                   self.observable_occ_total / self.occ_total))

        print("#roi = {}; #roi/#total = {}".format(self.roi_total, self.roi_total / (np.product(self.shape))))
        print("#occ = {}; #occ/#total = {}".format(self.occ_total, self.occ_total / (np.product(self.shape))))
        print("#free = {}; #free/#total = {}".format(self.free_total, self.free_total / (np.product(self.shape))))

    def compute_fov(self):
        axes = self.robot_rot.as_matrix().transpose()
        rh = np.radians(self.hfov / 2)
        rv = np.radians(self.vfov / 2)
        vec_left_down = (Rotation.from_rotvec(rh * axes[2]) * Rotation.from_rotvec(rv * axes[1])).apply(axes[0])
        vec_left_up = (Rotation.from_rotvec(rh * axes[2]) * Rotation.from_rotvec(-rv * axes[1])).apply(axes[0])
        vec_right_down = (Rotation.from_rotvec(-rh * axes[2]) * Rotation.from_rotvec(rv * axes[1])).apply(axes[0])
        vec_right_up = (Rotation.from_rotvec(-rh * axes[2]) * Rotation.from_rotvec(-rv * axes[1])).apply(axes[0])
        ep_left_down = self.robot_pos + vec_left_down * self.max_sensor_range
        ep_left_up = self.robot_pos + vec_left_up * self.max_sensor_range
        ep_right_down = self.robot_pos + vec_right_down * self.max_sensor_range
        ep_right_up = self.robot_pos + vec_right_up * self.max_sensor_range

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

    def generate_unknown_map_layer5(self, cam_pos):
        rot_vecs = self.compute_rot_vecs(self.obs_hrange[0], self.obs_hrange[1], self.obs_hsteps,
                                         self.obs_vrange[0], self.obs_vrange[1], self.obs_vsteps)
        dist = self.obs_drange[1]
        unknown_map = count_unknown_layer5_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos), rot_vecs,
                                                      1.0, dist)
        known_free_map = count_known_free_layer5_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos),
                                                            rot_vecs, 1.0, dist)
        known_occupied_map = count_known_occupied_layer5_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos),
                                                                    rot_vecs, 1.0, dist)
        known_target_map = count_known_target_layer5_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos),
                                                                rot_vecs, 1.0, dist)
        return unknown_map, known_free_map, known_occupied_map, known_target_map

    def generate_unknown_map_layer8(self, cam_pos):
        rot_vecs = self.compute_rot_vecs(self.obs_hrange[0], self.obs_hrange[1], self.obs_hsteps,
                                         self.obs_vrange[0], self.obs_vrange[1], self.obs_vsteps)
        dist = self.obs_drange[1]
        unknown_map = count_unknown_layer8_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos), rot_vecs,
                                                      1.0, dist)
        known_free_map = count_known_free_layer8_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos),
                                                            rot_vecs, 1.0, dist)
        known_occupied_map = count_known_occupied_layer8_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos),
                                                                    rot_vecs, 1.0, dist)
        known_target_map = count_known_target_layer8_vectorized(self.known_map, generate_vec3d_from_arr(cam_pos),
                                                                rot_vecs, 1.0, dist)

        return unknown_map, known_free_map, known_occupied_map, known_target_map

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

        self.known_map, roi_cells, occupied_cells, free_cells, coords, values = field_env_3d_helper.update_grid_inds_in_view(
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
                ep_right_up)),

            self.hrays,
            self.vrays
        )

        self.path_coords.append(cam_pos)

        if self.head:
            self.gui.messenger.send('update_fov_and_cells',
                                    [cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up,
                                     coords, values], 'default')
            self.gui.messenger.send('draw_coord_line',
                                    [self.path_coords], 'default')

            self.gui.gui_done.wait()
            self.gui.gui_done.clear()

        # print("Updating field took {} s".format(time.perf_counter() - time_start))
        return roi_cells, occupied_cells, free_cells

    def move_robot(self, direction):
        collision = False
        future_robot_pos = self.robot_pos + direction
        # future_robot_pos = np.clip(future_robot_pos, self.world_bound.lower_bound, self.world_bound.upper_bound)
        if ((in_plant_bounds(self.plant_bounding_boxes, future_robot_pos)) or
                out_of_world_bound(self.world_bound, future_robot_pos) or
                out_of_sensor_position_bound(self.sensor_position_bound, future_robot_pos)):
            # do nothing, do not update robot_pose
            self.relative_position = np.zeros_like(direction)
            collision = True
        else:
            # update robot_pose
            self.relative_position = direction
            self.robot_pos = future_robot_pos

        if collision:
            self.collision_count += 1
        else:
            self.collision_count = 0
        return collision

    def rotate_robot(self, rot):
        self.robot_rot = rot * self.robot_rot
        self.relative_rotation = rot

    def step(self, action):
        # actions = [0, 2]
        # action = actions[self.step_count % 2]
        # if not self.parser_args.train:
        #     print("action:{}".format(action))
        # print(self.step_count,":", action)
        axes = self.robot_rot.as_matrix().transpose()
        relative_move, relative_rot = self.action_space.get_relative_move_rot(axes, action, self.MOVE_STEP,
                                                                              self.ROT_STEP)

        collision = self.move_robot(relative_move)
        self.rotate_robot(relative_rot)
        cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up = self.compute_fov()
        found_roi, found_occ, found_free = self.update_grid_inds_in_view(cam_pos, ep_left_down, ep_left_up,
                                                                         ep_right_down, ep_right_up)
        visit_gain, coverage_rate = self.update_visit_map()

        self.found_roi_sum += found_roi
        self.found_occ_sum += found_occ
        self.found_free_sum += found_free

        self.step_count += 1
        done = (self.found_roi_sum == self.roi_total) or (self.step_count >= self.max_steps)

        # 5 * 36 * 18
        unknown_map, known_free_map, known_occupied_map, known_target_map = self.generate_unknown_map_layer(cam_pos)

        # 20 * 36 * 18
        self.map = concat(unknown_map, known_free_map, known_occupied_map, known_target_map, np.uint8)

        reward = self.get_reward(visit_gain, found_free, found_occ, found_roi, collision)

        # step
        info = {"visit_gain": visit_gain, "new_free_cells": found_free, "new_occupied_cells": found_occ,
                "new_found_rois": found_roi, "reward": reward, "coverage_rate": coverage_rate, "collision": collision}

        inputs = self.get_inputs()

        save_observation_map(self.map, self.step_count, self.parser_args)
        return inputs, reward, done, info

    def generate_unknown_map_layer(self, cam_pos):
        if "obs_layers" in self.env_config.keys() and self.env_config["obs_layers"] == 8:
            unknown_map, known_free_map, known_occupied_map, known_target_map = self.generate_unknown_map_layer8(
                cam_pos)
        else:
            unknown_map, known_free_map, known_occupied_map, known_target_map = self.generate_unknown_map_layer5(
                cam_pos)
        return unknown_map, known_free_map, known_occupied_map, known_target_map

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

    def get_reward(self, visit_gain, found_free, found_occ, found_roi, collision):
        weight = self.training_config["rewards"]
        reward = weight["visit_gain_weight"] * visit_gain + \
                 weight["free_weight"] * found_free + \
                 weight["occ_weight"] * found_occ + \
                 weight["roi_weight"] * found_roi + \
                 weight["collision_weight"] * max(self.collision_count - 1, 0)
        # print(self.step_count, "collision:{}; reward:{}".format(collision, reward))

        if reward == 0:
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        reward += weight["stuck_weight"] * max(self.stuck_count - 5, 0)

        return reward

    def update_visit_map(self):
        # to encourage exploration
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
        self.visit_map[start_x: end_x, start_y: end_y, start_z: end_z] = np.ones_like(neighbor_visit_map).astype(
            np.uint8)
        coverage_rate = np.sum(self.visit_map) / np.product(self.visit_shape)
        return visit_gain, coverage_rate

    def reset(self):
        self.reset_count += 1
        self.initialize()

        if self.head:
            self.gui.messenger.send('reset', [], 'default')
            self.gui.gui_done.wait()
            self.gui.gui_done.clear()
            # self.gui.reset()
        cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up = self.compute_fov()
        self.update_grid_inds_in_view(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)

        unknown_map, known_free_map, known_occupied_map, known_target_map = self.generate_unknown_map_layer(cam_pos)
        self.map = concat(unknown_map, known_free_map, known_occupied_map, known_target_map, np.uint8)
        # map = make_up_8x15x9x9_map(map)
        return self.get_inputs(), {}
