import numpy as np
from field_env_3d import Field, Action


class RandomAgent:
    def __init__(self, env):
        if type(env) is not Field:
            raise TypeError("Environment should be of type Field.")
        self.env = env

    def get_action(self, observed_map, robot_pose):
        return np.random.choice(list(Action))

    def store_reward(self, reward, done):
        return

    def reset(self):
        return