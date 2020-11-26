import pygame as pg
from field_env import Field, Action
from human_player import HumanPlayer

pg.init()

field = Field(shape=(64, 64), target_count=100, sensor_range=5, scale=8)
player = HumanPlayer(field)

while True:
    field.draw_field()
    action = player.get_action()
    reward = field.step(action)
    print(reward)

pg.quit()
