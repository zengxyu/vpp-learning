import numpy as np


class RobotPoseRecord:
    def __init__(self, capacity=50000, robot_pose_shape=(7,), max_episode=1000):
        self.capacity = capacity
        self.max_episode = max_episode
        self.cursor = 0
        self.size = 0
        self.robot_poses = np.zeros((self.capacity, robot_pose_shape[0]))
        self.robot_pose_mean = np.zeros(robot_pose_shape)
        self.robot_pose_std = np.zeros(robot_pose_shape)

    def add_robot_pose(self, robot_pose):
        self.robot_poses[self.cursor] = robot_pose
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if self.cursor % self.max_episode == 0:
            self.robot_pose_mean = np.mean(self.robot_poses[:self.size], axis=0)
            self.robot_pose_std = np.std(self.robot_poses[:self.size], axis=0)

    def get_robot_pose_mean(self):
        return self.robot_pose_mean

    def get_robot_pose_std(self):
        return self.robot_pose_std
