import numpy as np


class GridCellAccessRecord:
    def __init__(self, shape):
        self.shape = shape
        self.grid = np.zeros(shape)

    def get_reward_of_new_visit(self, robot_pose):
        reward = 0
        # print("robot_pose:\n", robot_pose)
        x = np.clip(int(robot_pose[0]), 0, self.shape[0] - 1)
        y = np.clip(int(robot_pose[1]), 0, self.shape[1] - 1)
        z = np.clip(int(robot_pose[2]), 0, self.shape[2] - 1)

        if self.grid[x, y, z] == 0:
            # visit a new cell, get reward 1
            reward = 1

        self.grid[x, y, z] += 1

        return reward

    def clear(self):
        self.grid = np.zeros(self.shape)

    def get_reward(self):
        count = np.count_nonzero(self.grid)
        return count
