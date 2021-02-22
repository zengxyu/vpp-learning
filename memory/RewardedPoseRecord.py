import numpy as np


class RewardedPoseRecord:
    def __init__(self, grid_shape=(256, 256, 256), shape=(10000, 3)):
        self.shape = shape
        self.cursor_index = 1
        self.record_size = 1
        self.rewarded_pose_records = np.zeros(shape)
        self.rewarded_pose_records[0] = np.array([grid_shape[0], grid_shape[1], grid_shape[2]]) / 2
        # print("\nmean pose:", self.get_pose_mean())
        # print("record_size:", self.record_size)

    def put_pose(self, robot_pose):
        self.rewarded_pose_records[self.cursor_index] = robot_pose[:3]
        self.cursor_index = (self.cursor_index + 1) % self.shape[0]
        self.record_size = min(self.record_size + 1, self.shape[0])
        # print("\nput pose")
        # print("mean pose:", self.get_pose_mean())
        # print("record_size:", self.record_size)

    def get_reward(self, robot_pose):
        # print("\nmean pose:", self.get_pose_mean())
        # print("record_size:", self.record_size)
        reward = -np.sqrt(np.sum(np.square(robot_pose[:3] - self.get_pose_mean())))
        return reward

    def get_pose_mean(self):
        return np.sum(self.rewarded_pose_records, axis=0) / self.record_size

    def clear(self):
        self.rewarded_pose_records = np.zeros(self.shape)
