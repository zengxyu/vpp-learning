#!/usr/bin/env python

import os

import numpy as np
import pygame as pg
from scipy import ndimage
from shapely import geometry
import cv2
import imutils
from enum import IntEnum

class Action(IntEnum):
    DO_NOTHING = 0,
    MOVE_FORWARD = 1,
    MOVE_BACKWARD = 2,
    MOVE_LEFT = 3,
    MOVE_RIGHT = 4,
    ROTATE_LEFT = 5,
    ROTATE_RIGHT = 6

class FieldValues(IntEnum):
    UNKNOWN = 0,
    FREE = 1,
    OCCUPIED = 2,
    TARGET = 3

class Field:
    def __init__(self, shape, target_count, sensor_range, scale, max_steps, headless = False):
        self.target_count = target_count
        self.found_targets = 0
        self.sensor_range = sensor_range
        self.shape = shape
        self.scale = scale
        self.max_steps = max_steps
        self.headless = headless
        self.ROTATION_LIST = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
        self.ROTATION_ANGLES = range(0, 360, 45)
        self.ROTATION_MATRICES = [[[np.cos(i), -np.sin(i)], [np.sin(i), np.cos(i)]] for i in np.deg2rad(self.ROTATION_ANGLES)]
        self.UNKNOWN_COLOR = 0x000000
        self.FREE_COLOR = 0x696969
        self.OCCUPIED_COLOR = 0xD3D3D3
        self.TARGET_COLOR = 0x008800
        self.ROBOT_COLOR = 0xFFA500
        self.COLOR_DICT = {FieldValues.UNKNOWN: self.UNKNOWN_COLOR, FieldValues.FREE: self.FREE_COLOR, FieldValues.OCCUPIED: self.OCCUPIED_COLOR, FieldValues.TARGET: self.TARGET_COLOR}
        if not headless:
            self.screen = pg.display.set_mode((shape[0]*scale*2, shape[1]*scale), 0, 32)
            self.global_mat = pg.Surface((shape[0]*scale, shape[1]*scale), 0, 32)
            self.pov_mat = pg.Surface((shape[0]*scale, shape[1]*scale), 0, 32)
            self.font = pg.font.SysFont(pg.font.get_default_font(), 30)

    def apply_mask(self, image, mask, color, alpha=0.2):
        image = np.where(mask, (1 - alpha) * image + alpha * mask * color, image)
        return image

    #def apply_cropped_mask(self, image, box, mask, color, alpha=0.5):
    #    x1, y1, x2, y2 = box
    #    for c in range(3):
    #        image[y1:y2, x1:x2, c] = np.where(mask == 1, (1 - alpha) * image[y1:y2, x1:x2, c] + alpha * mask * color[c], image[y1:y2, x1:x2, c])
    #    return image

    def transform_to_pov(self, field, scale = 1):
        shifted = ndimage.shift(field, (np.asarray(self.field.shape) / 2 - self.robot_pos) * scale, order=0)
        rotated = ndimage.rotate(shifted, -self.ROTATION_ANGLES[self.robot_rotind], reshape=False, order=0)
        #rotated = imutils.rotate(np.float32(shifted), -self.ROTATION_ANGLES[self.robot_rotind])
        #rotated = ndimage.affine_transform(shifted, self.ROTATION_MATRICES[self.robot_rotind], order=0)
        return rotated

    def colorize_and_upscale(self, field, scale, obsarea = None, fovarray = None):
        sfield = np.vectorize(self.COLOR_DICT.__getitem__)(field)
        if obsarea is not None:
            sfield = self.apply_mask(sfield, obsarea, self.ROBOT_COLOR)
        if fovarray is not None:
            sfield = self.apply_mask(sfield, fovarray, self.ROBOT_COLOR)

        sfield = np.kron(sfield, np.ones((scale, scale)))
        return sfield

    def draw_arrow(self, sfield, scale, spos, rot):
        cv2.arrowedLine(sfield, (spos[1] + scale//2 * (-rot[1] + 1) , spos[0] + scale//2 * (-rot[0] + 1)), (spos[1] + scale//2 * (rot[1] + 1), spos[0] + scale//2 * (rot[0] + 1)), self.ROBOT_COLOR)
    
    def draw_field(self):
        sfield = self.colorize_and_upscale(self.field, self.scale, self.obsarea, self.fovarray)
        #pov_field, pov_obsarea, pov_fovarray = self.transform_to_pov(self.field), self.transform_to_pov(self.obsarea), self.transform_to_pov(self.fovarray)
        s_pov_field = self.colorize_and_upscale(self.observed_field, self.scale)
        spos = self.robot_pos * self.scale
        rot = self.ROTATION_LIST[self.robot_rotind]
        self.draw_arrow(sfield, self.scale, spos, rot)
        self.draw_arrow(s_pov_field, self.scale, spos, rot) #np.asarray(s_pov_field.shape) // 2, (0, -1)
        pg.surfarray.blit_array(self.global_mat, sfield)
        self.screen.blit(self.global_mat, (0, 0))
        #rotated_screen = self.transform_to_pov(sfield, self.scale) #ndimage.rotate(sfield, 45, reshape=False, order=0)
        pg.surfarray.blit_array(self.pov_mat, s_pov_field)
        self.screen.blit(self.pov_mat, (sfield.shape[0], 0))
        top_text = self.font.render('Step {} / {}, {} / {} targets found'.format(self.step_count, self.max_steps, self.found_targets, self.target_count), False, (255, 255, 255))
        self.screen.blit(top_text, (5, 5))
        pg.display.update()

    #def point_in_triangle(self, p, v1, v2, v3):
    #    c1 = (v2[0] - v1[0]) * (p[1] - v1[1]) - (v2[1] - v1[1]) * (p[0] - v1[0])
    #    c2 = (v3[0] - v2[0]) * (p[1] - v2[1]) - (v3[1] - v2[1]) * (p[0] - v2[0])
    #    c3 = (v1[0] - v3[0]) * (p[1] - v3[1]) - (v1[1] - v3[1]) * (p[0] - v3[0])
    #    return ((c1<0 and c2<0 and c3<0) or (c1>0 and c2>0 and c3>0))

    def get_bb_points(self, v1, v2, v3):
        clip = lambda x, l, u: max(l, min(u, x))
        minx = clip(min(v1[0], v2[0], v3[0]), 0, self.field.shape[0] - 1)
        maxx = clip(max(v1[0], v2[0], v3[0]), 0, self.field.shape[0] - 1)
        miny = clip(min(v1[1], v2[1], v3[1]), 0, self.field.shape[1] - 1)
        maxy = clip(max(v1[1], v2[1], v3[1]), 0, self.field.shape[1] - 1)
        return [minx, miny], [maxx, maxy]

    def compute_fov(self):
        rot = self.ROTATION_LIST[self.robot_rotind]
        rabs = np.sqrt(rot[0]**2 + rot[1]**2)
        rpos = self.robot_pos
        rleft = (self.robot_rotind - 1) % len(self.ROTATION_LIST)
        rright = (self.robot_rotind + 1) % len(self.ROTATION_LIST)
        rlpos = np.rint(rpos + np.asarray(self.ROTATION_LIST[rleft]) * self.sensor_range * rabs).astype(int)
        rrpos = np.rint(rpos + np.asarray(self.ROTATION_LIST[rright]) * self.sensor_range * rabs).astype(int)
        fovpoly = geometry.Polygon([rpos, rlpos, rrpos])
        bbmin, bbmax = self.get_bb_points(rpos, rlpos, rrpos)
        xrange = range(bbmin[0], bbmax[0] + 1)
        yrange = range(bbmin[1], bbmax[1] + 1)
        self.fovarray = np.zeros(self.field.shape, dtype=bool)
        for x in xrange:
            for y in yrange:
                if fovpoly.intersects(geometry.Point([x, y])):
                    self.fovarray[x, y] = True
                    self.obsarea[x, y] = True
                    if self.observed_field[x, y] == FieldValues.UNKNOWN and self.field[x, y] == FieldValues.TARGET:
                        self.found_targets += 1

                    self.observed_field[x, y] = self.field[x, y]

        #self.found_targets = np.count_nonzero(np.logical_and(self.obsarea, self.field))

        #barr = fovpoly.contains(points)
        #print(barr)
        #grid = np.mgrid[bbmin[0]:bbmax[0], bbmin[1]:bbmax[1]]
        #print(grid)
        #barr = np.where(self.point_in_triangle(grid, rpos, rlpos, rrpos))
        #print(barr)

    def move_robot(self, direction):
        self.robot_pos += direction
        for i in range(2):
            if self.robot_pos[i] < 0:
                self.robot_pos[i] = 0
            elif self.robot_pos[i] >= self.field.shape[i]:
                self.robot_pos[i] = self.field.shape[i] - 1

    def rotate_robot(self, direction):
        self.robot_rotind = (self.robot_rotind + direction) % len(self.ROTATION_LIST)

    def step(self, action):
        if action == Action.MOVE_FORWARD:
            self.move_robot(self.ROTATION_LIST[self.robot_rotind])
        elif action == Action.MOVE_BACKWARD:
            self.move_robot(self.ROTATION_LIST[(self.robot_rotind + 4) % len(self.ROTATION_LIST)])
        elif action == Action.MOVE_LEFT:
            self.move_robot(self.ROTATION_LIST[(self.robot_rotind - 2) % len(self.ROTATION_LIST)])
        elif action == Action.MOVE_RIGHT:
            self.move_robot(self.ROTATION_LIST[(self.robot_rotind + 2) % len(self.ROTATION_LIST)])
        elif action == Action.ROTATE_LEFT:
            self.rotate_robot(-1)
        elif action == Action.ROTATE_RIGHT:
            self.rotate_robot(1)

        targets_before = self.found_targets
        self.compute_fov()
        if not self.headless:
            self.draw_field()

        robot_pose = [self.robot_pos[0], self.robot_pos[1], np.deg2rad(self.ROTATION_ANGLES[self.robot_rotind])]
        reward = self.found_targets - targets_before
        self.step_count += 1
        done = (self.found_targets == self.target_count) or (self.step_count >= self.max_steps)
        return self.observed_field, robot_pose, reward, done

    def reset(self):
        #self.field = np.full(self.shape, FieldValues.FREE, dtype=FieldValues)
        #i = np.random.random_integers(0, self.shape[0] - 1, self.target_count)
        #j = np.random.random_integers(0, self.shape[1] - 1, self.target_count)
        #self.field[i, j] = FieldValues.TARGET
        size = np.product(self.shape)
        self.field = np.full(size, FieldValues.FREE, dtype=FieldValues)
        i = np.random.choice(size, self.target_count)
        self.field[i] = FieldValues.TARGET
        self.field = self.field.reshape(self.shape)
        self.robot_pos = np.array([np.random.randint(self.shape[0]), np.random.randint(self.shape[1])])
        self.robot_rotind = np.random.randint(len(self.ROTATION_LIST))
        self.fovarray = np.zeros(self.field.shape, dtype=bool)
        self.obsarea = np.zeros(self.field.shape, dtype=bool)
        self.observed_field = np.zeros(self.field.shape, dtype=np.uint32)
        self.step_count = 0
        self.found_targets = 0

        self.compute_fov()

        if not self.headless:
            self.draw_field()

        return self.observed_field, [self.robot_pos[0], self.robot_pos[1], np.deg2rad(self.ROTATION_ANGLES[self.robot_rotind])]