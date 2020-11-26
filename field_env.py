#!/usr/bin/env python

import os

import numpy as np
import pygame as pg
from scipy import ndimage
from shapely import geometry
import cv2
import imutils
from enum import Enum

class Action(Enum):
    DO_NOTHING = 0,
    MOVE_FORWARD = 1,
    MOVE_BACKWARD = 2,
    MOVE_LEFT = 3,
    MOVE_RIGHT = 4,
    ROTATE_LEFT = 5,
    ROTATE_RIGHT = 6

class Field:
    def __init__(self, shape, target_count, sensor_range, scale):
        self.field = np.zeros(shape, dtype=np.uint32)
        self.target_count = target_count
        self.found_targets = 0
        self.sensor_range = sensor_range
        self.scale = scale
        i = np.random.random_integers(0, shape[0] - 1, target_count)
        j = np.random.random_integers(0, shape[1] - 1, target_count)
        self.field[i, j] = 1
        self.ROTATION_LIST = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
        self.ROTATION_ANGLES = range(0, 360, 45)
        self.ROTATION_MATRICES = [[[np.cos(i), -np.sin(i)], [np.sin(i), np.cos(i)]] for i in np.deg2rad(self.ROTATION_ANGLES)]
        self.TARGET_COLOR = 0x008800
        self.ROBOT_COLOR = 0xFFA500
        self.robot_pos = np.random.randint(63, size=2)
        self.robot_rotind = np.random.randint(len(self.ROTATION_LIST))
        self.fovarray = np.zeros(self.field.shape, dtype=bool)
        self.obsarea = np.zeros(self.field.shape, dtype=bool)
        self.observerd_field = np.zeros(self.field.shape, dtype=np.uint32)
        sfield = np.kron(self.field, np.ones((scale, scale)))
        self.screen = pg.display.set_mode((sfield.shape[0]*2, sfield.shape[1]), 0, 32)
        self.global_mat = pg.Surface(sfield.shape, 0, 32)
        self.pov_mat = pg.Surface(sfield.shape, 0, 32)
        self.screen.blit(self.global_mat, (0, 0))
        self.screen.blit(self.pov_mat, (sfield.shape[0], 0))
        self.font = pg.font.SysFont(pg.font.get_default_font(), 30)
        score = self.font.render('{} / {} targets found'.format(self.found_targets, self.target_count), True, (255, 255, 255))
        self.screen.blit(score, (5, 5))
        pg.display.update()

    def compute_observerd_field(self):
        return

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

    def colorize_and_upscale(self, field, scale, obsarea, fovarray):
        sfield = np.where(field == 1, self.TARGET_COLOR, field)
        sfield = self.apply_mask(sfield, obsarea, self.ROBOT_COLOR)
        sfield = self.apply_mask(sfield, fovarray, self.ROBOT_COLOR)
        sfield = np.kron(sfield, np.ones((scale, scale)))
        return sfield

    def draw_arrow(self, sfield, scale, spos, rot):
        cv2.arrowedLine(sfield, (spos[1] + scale//2 * (-rot[1] + 1) , spos[0] + scale//2 * (-rot[0] + 1)), (spos[1] + scale//2 * (rot[1] + 1), spos[0] + scale//2 * (rot[0] + 1)), self.ROBOT_COLOR)
    
    def draw_field(self):
        sfield = self.colorize_and_upscale(self.field, self.scale, self.obsarea, self.fovarray)
        pov_field, pov_obsarea, pov_fovarray = self.transform_to_pov(self.field), self.transform_to_pov(self.obsarea), self.transform_to_pov(self.fovarray)
        s_pov_field = self.colorize_and_upscale(pov_field, self.scale, pov_obsarea, pov_fovarray)
        spos = self.robot_pos * self.scale
        rot = self.ROTATION_LIST[self.robot_rotind]
        self.draw_arrow(sfield, self.scale, spos, rot)
        self.draw_arrow(s_pov_field, self.scale, np.asarray(s_pov_field.shape) // 2, (0, -1))
        pg.surfarray.blit_array(self.global_mat, sfield)
        self.screen.blit(self.global_mat, (0, 0))
        #rotated_screen = self.transform_to_pov(sfield, self.scale) #ndimage.rotate(sfield, 45, reshape=False, order=0)
        pg.surfarray.blit_array(self.pov_mat, s_pov_field)
        self.screen.blit(self.pov_mat, (sfield.shape[0], 0))
        score = self.font.render('{} / {} targets found'.format(self.found_targets, self.target_count), False, (255, 255, 255))
        self.screen.blit(score, (5, 5))
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

        self.found_targets = np.count_nonzero(np.logical_and(self.obsarea, self.field))

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

    #def rotate_clockwise(self, rot_ind):

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
        reward = self.found_targets - targets_before
        return reward