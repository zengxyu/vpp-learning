#!/usr/bin/env python

import os

import numpy as np
import pygame as pg
import cv2
from pygame import surfarray

class Field:
    def __init__(self, shape, target_count, scale):
        size = np.product(shape)
        self.field = np.zeros(size, dtype=np.int)
        self.target_count = target_count
        self.scale = scale
        i = np.random.choice(np.arange(size), target_count)
        self.field[i] = 0xFFFFFFFF
        self.field = self.field.reshape(shape)
        self.ROTATION_LIST = ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))
        self.robot_pos = np.random.randint(63, size=2)
        self.robot_rotind = np.random.randint(len(self.ROTATION_LIST))
        sfield = np.kron(self.field, np.ones((scale, scale)))
        self.screen = pg.display.set_mode(sfield.shape[:2], 0, 32)
    
    def draw_field(self):
        s = self.scale
        sfield = np.kron(self.field, np.ones((s, s)))
        spos = self.robot_pos * s
        rot = self.ROTATION_LIST[self.robot_rotind]
        cv2.arrowedLine(sfield, (spos[1] + s//2 * (-rot[1] + 1) , spos[0] + s//2 * (-rot[0] + 1)), (spos[1] + s//2 * (rot[1] + 1), spos[0] + s//2 * (rot[0] + 1)), 0xFFA500)
        surfarray.blit_array(self.screen, sfield)
        pg.display.flip()

    def move_robot(self, direction):
        self.robot_pos += direction
        for i in range(2):
            if self.robot_pos[i] < 0:
                self.robot_pos[i] = 0
            elif self.robot_pos[i] >= self.field.shape[i]:
                self.robot_pos[i] = self.field.shape[i] - 1

    def rotate_robot(self, clockwise):
        if clockwise:
            self.robot_rotind -= 1
            if self.robot_rotind < 0:
                self.robot_rotind = len(self.ROTATION_LIST) - 1
        else:
            self.robot_rotind = (self.robot_rotind + 1) % len(self.ROTATION_LIST)

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
                self.rotate_robot(clockwise=True)
            elif e.key == pg.K_q:
                self.rotate_robot(clockwise=False)
        elif e.type == pg.QUIT:
            raise SystemExit()

def main():

    pg.init()

    field = Field((64, 64), 16, 8)

    while True:
        field.draw_field()
        field.wait_for_input()

    pg.quit()


if __name__ == "__main__":
    main()
