import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pickle

from utilities.util import get_project_path

max_len = 500
colors = ["#ff9933", "#669900", "#ff6666", "#6666ff", "#009999"]


def plot_reward(paths, labels, save_path):
    for i, path in enumerate(paths):
        loss, reward = pickle.load(open(path, 'rb'))
        reward = reward[:max_len]
        xs = np.arange(0, len(reward), 1)
        ys = reward
        plt.plot(xs, ys, colors[i], label=labels[i])
        plt.xlabel("x")
        plt.ylabel("y")
    # plt.ylim(0, 1)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

total_num = 75370
def plot_smooth_reward(paths, labels, save_path):
    smooth_every_n = 100
    for i, path in enumerate(paths):
        loss, reward = pickle.load(open(path, 'rb'))
        reward = reward[:max_len]
        xs = np.arange(0, len(reward), 1)
        ys = []
        for j in range(len(reward)):
            smooth_r = np.mean(reward[max(0, j - smooth_every_n):j])
            smooth_r = smooth_r / total_num
            ys.append(smooth_r)
        plt.plot(xs, ys, colors[i], label=labels[i], linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Percentage")
    # plt.ylim(0, 1)
    plt.legend(loc="upper left")
    plt.savefig(save_path)
    plt.show()


def plot_static_env():
    parent_dir = "/media/zeng/Workspace/results_paper/"
    path1 = os.path.join(parent_dir, "p3d_static_env_step_len_10_action_36", "loss_reward", "loss_reward.obj")
    path2 = os.path.join(parent_dir, "p3d_static_env_step_len_10_action_108", "loss_reward", "loss_reward.obj")
    paths = [path1, path2]
    labels = ["step length = 10", "step length = 5"]
    save_path = os.path.join(get_project_path(), "plot", "p3d_plot_images", "p3d_rewards")

    # plot_reward(paths, labels, save_path)
    plot_smooth_reward(paths, labels, save_path)


def plot_random_env():
    parent_dir = "/media/zeng/Workspace/results_paper/"
    path1 = os.path.join(parent_dir,
                         "p3d_random_env_seq_len_10_action_36_adaptive_1.2_reward_weighted_sum_of_targets_0.0006_unknowns",
                         "loss_reward", "loss_reward.obj")
    path2 = os.path.join(parent_dir,
                         "p3d_random_env_seq_len_10_action_108_adaptive_1.2_reward(weighted_sum_of_targets_0.008_unknowns)",
                         "loss_reward", "loss_reward.obj")
    path3 = os.path.join(parent_dir,
                         "p3d_random_env_seq_len_05_action_108_adaptive_1.1_reward(weighted_sum_of_targets_unknown_known)",
                         "loss_reward", "loss_reward.obj")
    path4 = os.path.join(parent_dir,
                         "p3d_random_env_random_action",
                         "loss_reward", "loss_reward.obj")
    paths = [path1, path2, path3, path4]
    labels = ["p3d_static_env_step_len_10_action_36", "p3d_static_env_step_len_10_action_108",
              "p3d_static_env_step_len_5_action_108", "p3d_random_env_random_action"]
    save_path = os.path.join(get_project_path(), "plot", "p3d_plot_images", "p3d_rewards_random_env")

    # plot_reward(paths, labels, save_path)
    plot_smooth_reward(paths, labels, save_path)


if __name__ == '__main__':
    # plot_static_env()
    plot_random_env()
