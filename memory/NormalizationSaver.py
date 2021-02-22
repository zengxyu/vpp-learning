import os
import pickle

import numpy as np


class NormalizationSaver:
    observed_maps = []
    robot_poses = []
    rewards1 = []
    rewards2 = []

    def __init__(self):
        pass

    def store_state_and_reward(self, observed_map, robot_pose, reward1, reward2, done):
        self.observed_maps.append(observed_map)
        self.robot_poses.append(robot_pose)
        self.rewards1.append(reward1)
        self.rewards2.append(reward2)

    def save_2_local(self, config_dir):
        observed_map_mean = np.mean(self.observed_maps, axis=0)
        observed_map_std = np.std(self.observed_maps, axis=0)

        robot_pose_mean = np.mean(self.robot_poses, axis=0)
        robot_pose_std = np.std(self.robot_poses, axis=0)

        reward_mean1 = np.mean(self.rewards1, axis=0)
        reward_std1 = np.std(self.rewards1, axis=0)

        reward_mean2 = np.mean(self.rewards2, axis=0)
        reward_std2 = np.std(self.rewards2, axis=0)

        # print("observed_map_mean shape:", observed_map_mean.shape)
        # print("observed_map_std std:", observed_map_std.shape)
        #
        # print("robot_pose_mean shape:", robot_pose_mean.shape)
        # print("robot_pose_std std:", robot_pose_std.shape)
        #
        # print("reward_mean shape:", reward_mean1.shape)
        # print("reward_std std:", reward_std1.shape)
        with open(os.path.join(config_dir, "observed_map_mean_std.pkl"), 'wb') as f:
            pickle.dump([observed_map_mean, observed_map_std], f)

        with open(os.path.join(config_dir, "robot_pose_mean_std.pkl"), 'wb') as f:
            pickle.dump([robot_pose_mean, robot_pose_std], f)

        with open(os.path.join(config_dir, "reward_mean_std1.pkl"), 'wb') as f:
            pickle.dump([reward_mean1, reward_std1], f)

        with open(os.path.join(config_dir, "reward_mean_std2.pkl"), 'wb') as f:
            pickle.dump([reward_mean2, reward_std2], f)


if __name__ == '__main__':
    config_dir = "config_dir"
    with open(os.path.join("..", config_dir, "observed_map_mean_std.pkl"), 'rb') as f:
        [observed_map_mean, observed_map_std] = pickle.load(f)

    with open(os.path.join("..", config_dir, "robot_pose_mean_std.pkl"), 'rb') as f:
        [robot_pose_mean, robot_pose_std] = pickle.load(f)
        print("\nrobot_pose_mean:")
        print(robot_pose_mean)

    with open(os.path.join("..", config_dir, "reward_mean_std1.pkl"), 'rb') as f:
        [reward_mean1, reward_std1] = pickle.load(f)
        print("\nreward_mean1:")
        print(reward_mean1)

    with open(os.path.join("..", config_dir, "reward_mean_std2.pkl"), 'rb') as f:
        [reward_mean2, reward_std2] = pickle.load(f)
        print("\nreward_mean2:")
        print(reward_mean2)
