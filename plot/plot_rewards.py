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


def plot_smooth_reward(paths, labels, save_path):
    smooth_every_n = 100
    for i, path in enumerate(paths):
        loss, reward = pickle.load(open(path, 'rb'))
        reward = reward[:max_len]
        xs = np.arange(0, len(reward), 1)
        ys = []
        for j in range(len(reward)):
            smooth_r = np.mean(reward[max(0, j - smooth_every_n):j])
            ys.append(smooth_r)
        plt.plot(xs, ys, colors[i], label=labels[i], linewidth=2)
        plt.xlabel("x")
        plt.ylabel("y")
    # plt.ylim(0, 1)
    plt.legend(loc="upper left")
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    parent_dir = "/media/zeng/Workspace/results_paper/"
    path1 = os.path.join(parent_dir, "p3d_static_env_step_len_10_action_36", "loss_reward", "loss_reward.obj")
    path2 = os.path.join(parent_dir, "p3d_static_env_step_len_10_action_108", "loss_reward", "loss_reward.obj")
    paths = [path1, path2]
    labels = ["step_len_10_action_36", "step_len_10_action_108"]
    save_path = os.path.join(get_project_path(), "plot", "p3d_plot_images", "p3d_rewards")

    # plot_reward(paths, labels, save_path)
    plot_smooth_reward(paths, labels, save_path)
