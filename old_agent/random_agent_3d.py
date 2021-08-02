import numpy as np

from environment.field_env_3d_unknown_map import Action


class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.observed_maps = []
        self.robot_poses = []
        self.rewards = []

    def get_action(self, observed_map, robot_pose):
        return np.random.choice(list(Action))

    def reset(self):
        return
