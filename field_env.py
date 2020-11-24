#!/usr/bin/env python

import os

import numpy as np
import pygame as pg
from shapely import geometry
import cv2

class Field:
    def __init__(self, shape, target_count, scale):
        size = np.product(shape)
        self.field = np.zeros(size, dtype=np.uint32)
        self.target_count = target_count
        self.scale = scale
        i = np.random.choice(np.arange(size), target_count)
        self.field[i] = 1
        self.field = self.field.reshape(shape)
        self.ROTATION_LIST = ((1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0))
        #self.ROTATION_LIST = ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))
        self.TARGET_COLOR = 0x008800
        self.ROBOT_COLOR = 0xFFA500
        self.SENSOR_RANGE = 5
        self.robot_pos = np.random.randint(63, size=2)
        self.robot_rotind = np.random.randint(len(self.ROTATION_LIST))
        self.fovarray = np.zeros(self.field.shape, dtype=bool)
        self.obsarea = np.zeros(self.field.shape, dtype=bool)
        sfield = np.kron(self.field, np.ones((scale, scale)))
        self.screen = pg.display.set_mode(sfield.shape[:2], 0, 32)

    def apply_mask(self, image, mask, color, alpha=0.2):
        image = np.where(mask, (1 - alpha) * image + alpha * mask * color, image)
        return image

    #def apply_cropped_mask(self, image, box, mask, color, alpha=0.5):
    #    x1, y1, x2, y2 = box
    #    for c in range(3):
    #        image[y1:y2, x1:x2, c] = np.where(mask == 1, (1 - alpha) * image[y1:y2, x1:x2, c] + alpha * mask * color[c], image[y1:y2, x1:x2, c])
    #    return image

    
    def draw_field(self):
        s = self.scale
        sfield = self.field.copy()
        sfield = np.where(sfield == 1, self.TARGET_COLOR, sfield)
        sfield = self.apply_mask(sfield, self.obsarea, self.ROBOT_COLOR)
        sfield = self.apply_mask(sfield, self.fovarray, self.ROBOT_COLOR)
        sfield = np.kron(sfield, np.ones((s, s)))
        spos = self.robot_pos * s
        rot = self.ROTATION_LIST[self.robot_rotind]
        cv2.arrowedLine(sfield, (spos[1] + s//2 * (-rot[1] + 1) , spos[0] + s//2 * (-rot[0] + 1)), (spos[1] + s//2 * (rot[1] + 1), spos[0] + s//2 * (rot[0] + 1)), self.ROBOT_COLOR)
        pg.surfarray.blit_array(self.screen, sfield)
        pg.display.flip()

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
        rlpos = np.rint(rpos + np.asarray(self.ROTATION_LIST[rleft]) * self.SENSOR_RANGE * rabs).astype(int)
        rrpos = np.rint(rpos + np.asarray(self.ROTATION_LIST[rright]) * self.SENSOR_RANGE * rabs).astype(int)
        print("Triangle points: {}, {}, {}".format(rpos, rlpos, rrpos))
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

    def wait_for_input(self):
        e = pg.event.wait()
        if e.type == pg.KEYDOWN:
            if e.key == pg.K_LEFT:
                self.move_robot((-1, 0))
            elif e.key == pg.K_RIGHT:
                self.move_robot((1, 0))
            elif e.key == pg.K_UP:
                self.move_robot((0, -1))
            elif e.key == pg.K_DOWN:
                self.move_robot((0, 1))
            elif e.key == pg.K_e:
                self.rotate_robot(1)
            elif e.key == pg.K_q:
                self.rotate_robot(-1)
        elif e.type == pg.QUIT:
            raise SystemExit()

def main():

    pg.init()

    field = Field((64, 64), 16, 8)

    while True:
        field.draw_field()
        field.wait_for_input()
        field.compute_fov()

    pg.quit()


if __name__ == "__main__":
    main()
