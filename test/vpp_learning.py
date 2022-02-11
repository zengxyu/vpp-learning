import pygame as pg
from environment.field_2d import Field
from test.human_player import HumanPlayer

pg.init()

field = Field(shape=(64, 64), target_count=100, sensor_range=5, scale=8, max_steps = 200)
#player = RandomAgent(field)
player = HumanPlayer(field)

observed_map, robot_pose = field.reset()
done = False

while not done:
    action = player.get_action(observed_map, robot_pose)
    observed_map, robot_pose, reward, done = field.step(action)
    print("Pose: {}, Reward: {}".format(robot_pose, reward))

pg.quit()
