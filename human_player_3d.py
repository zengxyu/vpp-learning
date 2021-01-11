import pygame as pg
from field_env_3d import Field, Action

class HumanPlayer:
    def __init__(self, env):
        if type(env) is not Field:
            raise TypeError("Environment should be of type Field.")
        self.env = env

    def get_action(self, observed_map, robot_pose):
        e = pg.event.wait()
        if e.type == pg.KEYDOWN:
            if e.key == pg.K_a:
                return Action.MOVE_LEFT
            elif e.key == pg.K_d:
                return Action.MOVE_RIGHT
            elif e.key == pg.K_w:
                return Action.MOVE_FORWARD
            elif e.key == pg.K_s:
                return Action.MOVE_BACKWARD
            elif e.key == pg.K_e:
                return Action.MOVE_DOWN
            elif e.key == pg.K_q:
                return Action.MOVE_UP
            elif e.key == pg.K_j:
                return Action.ROTATE_YAW_N
            elif e.key == pg.K_l:
                return Action.ROTATE_YAW_P
            elif e.key == pg.K_i:
                return Action.ROTATE_PITCH_P
            elif e.key == pg.K_k:
                return Action.ROTATE_PITCH_N
            elif e.key == pg.K_o:
                return Action.ROTATE_ROLL_P
            elif e.key == pg.K_u:
                return Action.ROTATE_ROLL_N
            else:
                return Action.DO_NOTHING
        elif e.type == pg.QUIT:
            raise SystemExit()

        return Action.DO_NOTHING

    def reset(self):
        return

