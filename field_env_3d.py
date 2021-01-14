#!/usr/bin/env python

import numpy as np
from enum import IntEnum
from scipy.spatial.transform import Rotation
from panda3d.core import Geom, GeomVertexData, GeomTriangles, GeomVertexWriter, GeomVertexFormat, LineSegs


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


class Field:
    def __init__(self, shape, target_count, sensor_range, hfov, vfov, scale, max_steps, headless=False):
        self.target_count = target_count
        self.found_targets = 0
        self.sensor_range = sensor_range
        self.hfov = hfov
        self.vfov = vfov
        self.shape = shape
        self.scale = scale
        self.max_steps = max_steps
        self.headless = headless
        self.robot_pos = [0.0, 0.0, 0.0]
        self.robot_rot = Rotation.from_quat([0, 0, 0, 1])

        self.MOVE_STEP = 1.0
        self.ROT_STEP = 15.0
        self.UNKNOWN_COLOR = (0, 0, 0, 1)
        self.FREE_COLOR = (105/255, 105/255, 105/255, 1)
        self.OCCUPIED_COLOR = (211/255, 211/255, 211/255, 1)
        self.TARGET_COLOR = (0, 136/255, 0, 1)
        self.ROBOT_COLOR = (1, 165/255, 0, 1)
        self.FOV_ALPHA = 1.0 # not working
        self.FOV_UP_COLOR = (220/255, 20/255, 60/255, self.FOV_ALPHA) # Crimson red
        self.FOV_DOWN_COLOR = (199/255, 21/255, 133/255, self.FOV_ALPHA) # MediumVioletRed
        self.FOV_LEFT_COLOR = (255/255, 69/255, 0/255, self.FOV_ALPHA) # OrangeRed
        self.FOV_RIGHT_COLOR = (255/255, 215/255, 0/255, self.FOV_ALPHA) # Gold
        self.FOV_FRONT_COLOR = (218/255, 112/255, 214/255, self.FOV_ALPHA) # Orchid
        self.COLOR_DICT = {FieldValues.UNKNOWN: self.UNKNOWN_COLOR, FieldValues.FREE: self.FREE_COLOR, FieldValues.OCCUPIED: self.OCCUPIED_COLOR, FieldValues.TARGET: self.TARGET_COLOR}

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

    def reset(self):
        # self.field = np.full(self.shape, FieldValues.FREE, dtype=FieldValues)
        # i = np.random.random_integers(0, self.shape[0] - 1, self.target_count)
        # j = np.random.random_integers(0, self.shape[1] - 1, self.target_count)
        # self.field[i, j] = FieldValues.TARGET
        size = np.product(self.shape)
        self.field = np.full(size, FieldValues.FREE, dtype=FieldValues)
        i = np.random.choice(size, self.target_count)
        self.field[i] = FieldValues.TARGET
        self.field = self.field.reshape(self.shape)
        self.robot_pos = np.array([np.random.randint(self.shape[0]), np.random.randint(self.shape[1]), np.random.randint(self.shape[2])])
        self.robot_rot = np.array([self.ROTATION_ANGLES(np.random.randint(len(self.ROTATION_ANGLES))),
                                   self.ROTATION_ANGLES(np.random.randint(len(self.ROTATION_ANGLES))),
                                   self.ROTATION_ANGLES(np.random.randint(len(self.ROTATION_ANGLES)))])
        self.fovarray = np.zeros(self.field.shape, dtype=bool)
        self.obsarea = np.zeros(self.field.shape, dtype=bool)
        self.observed_field = np.zeros(self.field.shape, dtype=np.uint32)
        self.step_count = 0
        self.found_targets = 0

        self.compute_fov()

        if not self.headless:
            self.draw_field()

        return self.observed_field, [self.robot_pos[0], self.robot_pos[1], np.deg2rad(self.ROTATION_ANGLES[self.robot_rotind])]
