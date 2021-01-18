#!/usr/bin/env python

import numpy as np
from enum import IntEnum
from scipy.spatial.transform import Rotation
from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeomNode, Geom, GeomVertexData, GeomTriangles, GeomVertexWriter, GeomVertexFormat, LineSegs
from panda3d.core import LVecBase3i, PTA_int, PTA_float
from p3d_voxgrid import VoxelGrid
import binvox_rw
from direct.stdpy import threading
import time
import field_env_3d_helper
from field_env_3d_helper import Vec3D


class Action(IntEnum):
    DO_NOTHING = 0,
    MOVE_FORWARD = 1,
    MOVE_BACKWARD = 2,
    MOVE_LEFT = 3,
    MOVE_RIGHT = 4,
    MOVE_UP = 5,
    MOVE_DOWN = 6,
    ROTATE_ROLL_P = 7,
    ROTATE_ROLL_N = 8,
    ROTATE_PITCH_P = 9,
    ROTATE_PITCH_N = 10,
    ROTATE_YAW_P = 11,
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


class FieldGUI(ShowBase):

    def __init__(self, env, scale):
        ShowBase.__init__(self)
        self.env = env
        self.scale = scale

        # Color constants
        self.FOV_ALPHA = 1.0  # not working
        self.FOV_UP_COLOR = (220/255, 20/255, 60/255, self.FOV_ALPHA)  # Crimson red
        self.FOV_DOWN_COLOR = (199/255, 21/255, 133/255, self.FOV_ALPHA)  # MediumVioletRed
        self.FOV_LEFT_COLOR = (255/255, 69/255, 0/255, self.FOV_ALPHA)  # OrangeRed
        self.FOV_RIGHT_COLOR = (255/255, 215/255, 0/255, self.FOV_ALPHA)  # Gold
        self.FOV_FRONT_COLOR = (218/255, 112/255, 214/255, self.FOV_ALPHA)  # Orchid

        self.OCCUPIED_UNSEEN_COLOR = (220/255, 20/255, 60/255, 1.0)  # Crimson red
        self.TARGET_UNSEEN_COLOR = (199/255, 21/255, 133/255, 1.0)  # MediumVioletRed
        self.FREE_SEEN_COLOR = (255/255, 69/255, 0/255, 1.0)  # OrangeRed
        self.OCCUPIED_SEEN_COLOR = (255/255, 215/255, 0/255, 1.0)  # Gold
        self.TARGET_SEEN_COLOR = (218/255, 112/255, 214/255, 1.0)  # Orchid

        self.voxgrid_node = GeomNode("voxgrid")
        self.fov_node = None
        self.fov_node_path = None

        self.colors = PTA_float(self.OCCUPIED_UNSEEN_COLOR + self.TARGET_UNSEEN_COLOR + self.FREE_SEEN_COLOR +
                                self.OCCUPIED_SEEN_COLOR + self.TARGET_SEEN_COLOR)
        self.voxgrid = VoxelGrid(self.env.shape, self.colors, self.scale)

        self.field_border = self.create_edged_cube([0, 0, 0], np.asarray(self.env.global_map.shape) * self.scale)
        self.render.attachNewNode(self.field_border)

        self.voxgrid_node.addGeom(self.voxgrid.getGeom())

        self.render.attachNewNode(self.voxgrid_node)

        self.gui_done = threading.Event()

        self.accept('reset', self.reset)
        self.accept('update_cell', self.updateSeenCell)
        self.accept('update_fov', self.updateFov)
        self.accept('update_fov_and_cells', self.updateFovAndCells)

    def reset(self):
        gui_map = self.env.global_map - 1  # GUI map is shifted by one for unseen cells
        self.voxgrid.reset(PTA_int(gui_map.flatten().tolist()))
        self.gui_done.set()

    def create_edged_cube(self, min, max):
        lines = LineSegs()
        lines.moveTo(min[0], min[1], min[2])
        lines.drawTo(max[0], min[1], min[2])
        lines.drawTo(max[0], max[1], min[2])
        lines.drawTo(min[0], max[1], min[2])
        lines.drawTo(min[0], min[1], min[2])
        lines.drawTo(min[0], min[1], max[2])
        lines.drawTo(max[0], min[1], max[2])
        lines.drawTo(max[0], min[1], min[2])

        lines.moveTo(max[0], max[1], min[2])
        lines.drawTo(max[0], max[1], max[2])
        lines.drawTo(max[0], min[1], max[2])

        lines.moveTo(max[0], max[1], max[2])
        lines.drawTo(min[0], max[1], max[2])
        lines.drawTo(min[0], min[1], max[2])

        lines.moveTo(min[0], max[1], max[2])
        lines.drawTo(min[0], max[1], min[2])

        lines.setThickness(4)
        node = lines.create()
        return node

    def create_fov_geom(self, cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up):
        vertices = GeomVertexData('vertices', GeomVertexFormat.get_v3c4(), Geom.UHStatic)
        vertices.setNumRows(16)
        vertex = GeomVertexWriter(vertices, 'vertex')
        color = GeomVertexWriter(vertices, 'color')

        cam_pos_scaled = np.asarray(cam_pos) * self.scale
        ep_left_up_scaled = np.asarray(ep_left_up) * self.scale
        ep_left_down_scaled = np.asarray(ep_left_down) * self.scale
        ep_right_up_scaled = np.asarray(ep_right_up) * self.scale
        ep_right_down_scaled = np.asarray(ep_right_down) * self.scale

        # left (0-2)
        vertex.addData3(tuple(cam_pos_scaled))
        color.addData4(self.FOV_LEFT_COLOR)
        vertex.addData3(tuple(ep_left_up_scaled))
        color.addData4(self.FOV_LEFT_COLOR)
        vertex.addData3(tuple(ep_left_down_scaled))
        color.addData4(self.FOV_LEFT_COLOR)

        # up (3-5)
        vertex.addData3(tuple(cam_pos_scaled))
        color.addData4(self.FOV_UP_COLOR)
        vertex.addData3(tuple(ep_right_up_scaled))
        color.addData4(self.FOV_UP_COLOR)
        vertex.addData3(tuple(ep_left_up_scaled))
        color.addData4(self.FOV_UP_COLOR)

        # right (6-8)
        vertex.addData3(tuple(cam_pos_scaled))
        color.addData4(self.FOV_RIGHT_COLOR)
        vertex.addData3(tuple(ep_right_down_scaled))
        color.addData4(self.FOV_RIGHT_COLOR)
        vertex.addData3(tuple(ep_right_up_scaled))
        color.addData4(self.FOV_RIGHT_COLOR)

        # down (9-11)
        vertex.addData3(tuple(cam_pos_scaled))
        color.addData4(self.FOV_DOWN_COLOR)
        vertex.addData3(tuple(ep_left_down_scaled))
        color.addData4(self.FOV_DOWN_COLOR)
        vertex.addData3(tuple(ep_right_down_scaled))
        color.addData4(self.FOV_DOWN_COLOR)

        # front (12-15)
        vertex.addData3(tuple(ep_left_down_scaled))
        color.addData4(self.FOV_FRONT_COLOR)
        vertex.addData3(tuple(ep_left_up_scaled))
        color.addData4(self.FOV_FRONT_COLOR)
        vertex.addData3(tuple(ep_right_down_scaled))
        color.addData4(self.FOV_FRONT_COLOR)
        vertex.addData3(tuple(ep_right_up_scaled))
        color.addData4(self.FOV_FRONT_COLOR)

        geom = Geom(vertices)
        tri_prim = GeomTriangles(Geom.UHStatic)
        tri_prim.add_consecutive_vertices(0, 12)
        tri_prim.add_vertices(13, 15, 12)
        tri_prim.add_vertices(14, 12, 15)

        geom.add_primitive(tri_prim)
        return geom

    def create_fov_lines(self, cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up):
        cam_pos_scaled = np.asarray(cam_pos) * self.scale
        ep_left_up_scaled = np.asarray(ep_left_up) * self.scale
        ep_left_down_scaled = np.asarray(ep_left_down) * self.scale
        ep_right_up_scaled = np.asarray(ep_right_up) * self.scale
        ep_right_down_scaled = np.asarray(ep_right_down) * self.scale

        lines = LineSegs()
        lines.moveTo(tuple(cam_pos_scaled))
        lines.drawTo(tuple(ep_left_down_scaled))
        lines.draw_to(tuple(ep_left_up_scaled))
        lines.draw_to(tuple(cam_pos_scaled))
        lines.draw_to(tuple(ep_right_down_scaled))
        lines.draw_to(tuple(ep_right_up_scaled))
        lines.draw_to(tuple(cam_pos_scaled))

        lines.move_to(tuple(ep_left_down_scaled))
        lines.draw_to(tuple(ep_right_down_scaled))

        lines.move_to(tuple(ep_left_up_scaled))
        lines.draw_to(tuple(ep_right_up_scaled))

        lines.setThickness(4)
        node = lines.create()
        return node

    def updateSeenCell(self, coord):
        # seen values 3 higher than unseen
        self.voxgrid.updateValue(LVecBase3i(coord), self.env.global_map[coord] + 3)
        self.gui_done.set()

    def updateSeenCells(self, coords, values):
        self.voxgrid.updateValues(coords, values)
        self.gui_done.set()

    def updateFov(self, cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up):
        if self.fov_node_path:
            self.fov_node_path.removeNode()

        self.fov_node = self.create_fov_lines(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)
        self.fov_node_path = self.render.attachNewNode(self.fov_node)
        self.gui_done.set()

    def updateFovAndCells(self, cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up, coords, values):
        if self.fov_node_path:
            self.fov_node_path.removeNode()

        self.fov_node = self.create_fov_lines(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)
        self.fov_node_path = self.render.attachNewNode(self.fov_node)
        self.voxgrid.updateValues(coords, values)
        self.gui_done.set()


class Field:
    def __init__(self, shape, sensor_range, hfov, vfov, max_steps, init_file=None, headless=False, scale=0.05):
        self.found_targets = 0
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

        if init_file:
            self.read_env_from_file(init_file, scale)

    def read_env_from_file(self, filename, scale):
        with open(filename, 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
        self.global_map = np.transpose(model.data, (2, 0, 1)).astype(int)
        self.target_count = np.count_nonzero(self.global_map)
        self.found_targets = 0
        self.global_map += 1  # Shift: 1 - free, 2 - occupied/target
        self.shape = self.global_map.shape
        self.known_map = np.zeros(self.shape)

        if not self.headless:
            self.gui = FieldGUI(self, scale)

    def compute_fov(self):
        axes = self.robot_rot.as_matrix().transpose()
        rh = np.radians(self.hfov/2)
        rv = np.radians(self.vfov/2)
        vec_left_down = (Rotation.from_rotvec(rh * axes[2]) * Rotation.from_rotvec(rv * axes[1])).apply(axes[0])
        vec_left_up = (Rotation.from_rotvec(rh * axes[2]) * Rotation.from_rotvec(-rv * axes[1])).apply(axes[0])
        vec_right_down = (Rotation.from_rotvec(-rh * axes[2]) * Rotation.from_rotvec(rv * axes[1])).apply(axes[0])
        vec_right_up = (Rotation.from_rotvec(-rh * axes[2]) * Rotation.from_rotvec(-rv * axes[1])).apply(axes[0])
        ep_left_down = self.robot_pos + vec_left_down * self.sensor_range
        ep_left_up = self.robot_pos + vec_left_up * self.sensor_range
        ep_right_down = self.robot_pos + vec_right_down * self.sensor_range
        ep_right_up = self.robot_pos + vec_right_up * self.sensor_range
        return self.robot_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up

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
        self.known_map, found_targets, coords, values = field_env_3d_helper.update_grid_inds_in_view(self.known_map, self.global_map, Vec3D(*tuple(cam_pos)),
                                                                                                     Vec3D(*tuple(ep_left_down)), Vec3D(*tuple(ep_left_up)),
                                                                                                     Vec3D(*tuple(ep_right_down)), Vec3D(*tuple(ep_right_up)))

        if not self.headless:
            self.gui.messenger.send('update_fov_and_cells', [cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up,
                                                             coords, values], 'default')
            self.gui.gui_done.wait()
            self.gui.gui_done.clear()

        print("Updating field took {} s".format(time.perf_counter() - time_start))

        return found_targets

    def update_grid_inds_in_view_old(self, cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up):
        time_start = time.perf_counter()
        bb_min, bb_max = self.get_bb_points([cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up])
        bb_min, bb_max = np.clip(np.rint(bb_min), [0, 0, 0], self.shape).astype(int), np.clip(np.rint(bb_max), [0, 0, 0], self.shape).astype(int)
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
                    p_proj, rel_dist = self.line_plane_intersection(ep_right_down, plane_normal, cam_pos, (point - cam_pos))
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
            self.gui.messenger.send('update_fov_and_cells', [cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up, coords, values], 'default')
            # self.gui.messenger.send('update_fov', [cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up], 'default')
            self.gui.gui_done.wait()
            self.gui.gui_done.clear()
            # self.gui.updateFov(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)

        print("Updating field took {} s".format(time.perf_counter() - time_start))

        return found_targets

    def move_robot(self, direction):
        self.robot_pos += direction
        self.robot_pos = np.clip(self.robot_pos, [0, 0, 0], self.shape)

    def rotate_robot(self, axis, angle):
        rot = Rotation.from_rotvec(np.radians(angle) * axis)
        self.robot_rot = rot * self.robot_rot

    def step(self, action):        
        axes = self.robot_rot.as_matrix().transpose()

        if action == Action.MOVE_FORWARD:
            self.move_robot(axes[0] * self.MOVE_STEP)
        elif action == Action.MOVE_BACKWARD:
            self.move_robot(-axes[0] * self.MOVE_STEP)
        elif action == Action.MOVE_LEFT:
            self.move_robot(axes[1] * self.MOVE_STEP)
        elif action == Action.MOVE_RIGHT:
            self.move_robot(-axes[1] * self.MOVE_STEP)
        elif action == Action.MOVE_UP:
            self.move_robot(axes[2] * self.MOVE_STEP)
        elif action == Action.MOVE_DOWN:
            self.move_robot(-axes[2] * self.MOVE_STEP)
        elif action == Action.ROTATE_ROLL_P:
            self.rotate_robot(axes[0], self.ROT_STEP)
        elif action == Action.ROTATE_ROLL_N:
            self.rotate_robot(axes[0], -self.ROT_STEP)
        elif action == Action.ROTATE_PITCH_P:
            self.rotate_robot(axes[1], self.ROT_STEP)
        elif action == Action.ROTATE_PITCH_N:
            self.rotate_robot(axes[1], -self.ROT_STEP)
        elif action == Action.ROTATE_YAW_N:
            self.rotate_robot(axes[2], self.ROT_STEP)
        elif action == Action.ROTATE_YAW_P:
            self.rotate_robot(axes[2], -self.ROT_STEP)

        cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up = self.compute_fov()
        new_targets_found = self.update_grid_inds_in_view(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)
        self.found_targets += new_targets_found
        self.step_count += 1
        done = (self.found_targets == self.target_count) or (self.step_count >= self.max_steps)
        return self.known_map, np.concatenate((self.robot_pos, self.robot_rot.as_quat())), new_targets_found, done

    def reset(self):
        self.known_map = np.zeros(self.shape)
        self.observed_area = np.zeros(self.shape, dtype=bool)
        self.robot_pos = np.random.uniform((0.0, 0.0, 0.0), self.shape)
        self.robot_rot = Rotation.random()
        self.step_count = 0
        self.found_targets = 0

        if not self.headless:
            self.gui.messenger.send('reset', [], 'default')
            self.gui.gui_done.wait()
            self.gui.gui_done.clear()
            # self.gui.reset()

        cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up = self.compute_fov()
        self.update_grid_inds_in_view(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)

        print(self.robot_pos)
        print(self.robot_rot.as_quat())

        return self.known_map, np.concatenate((self.robot_pos, self.robot_rot.as_quat()))
