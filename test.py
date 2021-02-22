import os
import pickle

config_dir = "config_dir"

if __name__ == '__main__':
    with open(os.path.join(config_dir, "observed_map_mean_std.pkl"), 'rb') as f:
        observed_map_mean, observed_map_std = pickle.load(f)

    with open(os.path.join(config_dir, "robot_pose_mean_std.pkl"), 'rb') as f:
        robot_pose_mean, robot_pose_std = pickle.load(f)
        print("robot_pose_mean:{}".format(robot_pose_mean))

    with open(os.path.join(config_dir, "reward_mean_std.pkl"), 'rb') as f:
        reward_mean, reward_std = pickle.load(f)
        print("reward_mean:{}".format(reward_mean))

