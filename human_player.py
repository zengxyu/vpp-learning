import pygame as pg
from field_env import Field, Action

class HumanPlayer:
    def __init__(self, env):
        if type(env) is not Field:
            raise TypeError("Environment should be of type Field.")
        self.env = env

    def get_action(self, ob=None):
        e = pg.event.wait()
        if e.type == pg.KEYDOWN:
            if e.key == pg.K_LEFT:
                return Action.MOVE_LEFT
            elif e.key == pg.K_RIGHT:
                return Action.MOVE_RIGHT
            elif e.key == pg.K_UP:
                return Action.MOVE_FORWARD
            elif e.key == pg.K_DOWN:
                return Action.MOVE_BACKWARD
            elif e.key == pg.K_e:
                return Action.ROTATE_RIGHT
            elif e.key == pg.K_q:
                return Action.ROTATE_LEFT
            else:
                return Action.DO_NOTHING
        elif e.type == pg.QUIT:
            raise SystemExit()

        return Action.DO_NOTHING

    def reset(self):
        return

