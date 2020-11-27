import pygame as pg
from field_env import Field, Action
from human_player import HumanPlayer

pg.init()

field = Field(shape=(64, 64), target_count=100, sensor_range=5, scale=8)
player = HumanPlayer(field)

observed_map, robot_pose = field.reset()

while True:
    action = player.get_action(observed_map, robot_pose)
    observed_map, robot_pose, reward = field.step(action)
    print("Pose: {}, Reward: {}".format(robot_pose, reward))

pg.quit()
