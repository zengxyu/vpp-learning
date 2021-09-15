import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pickle

from utilities.util import get_project_path

max_len = 450
total_num = 75370

font_size = 14
colors = ["#ff6666", "#6666ff", "#009999", "#ff9933", "#669900"]
font1 = {
    'size': font_size,
}
font2 = {'size': font_size + 2,
         }


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
    plt.legend(prop=font1)
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
            smooth_r = smooth_r / total_num
            ys.append(smooth_r)
        plt.plot(xs, ys, colors[i], label=labels[i], linewidth=2)
    plt.xlabel("Episode", font2)
    plt.ylabel("Coverage rate", font2)

    plt.tick_params(labelsize=font_size)
    # plt.ylim(0, 1)
    plt.legend(loc="upper left", prop=font1)
    plt.subplots_adjust(bottom=0.128)
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
    in_parent_dir = "/media/zeng/Workspace/results_paper/out_p3d_final/training"
    path0 = os.path.join(in_parent_dir, "out_p3d_random_step_len_10_36_action_with_scheduler",
                         "loss_reward", "loss_reward.obj")
    path1 = os.path.join(in_parent_dir, "out_p3d_random_step_len_5_36_action", "loss_reward", "loss_reward.obj")
    path2 = os.path.join(in_parent_dir, "p3d_random_env_random_action", "loss_reward", "loss_reward.obj")

    paths = [path0, path1, path2]
    labels = ["Ours with sequence length = 10", "Ours with sequence length = 5", "Random exploration"]
    save_dir = os.path.join(get_project_path(), "plot", "p3d_training_plot")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(path0):
        print("Path:{} not exist!".format(path0))
    if not os.path.exists(path1):
        print("Path:{} not exist!".format(path1))
    if not os.path.exists(path2):
        print("Path:{} not exist!".format(path2))
    save_path = os.path.join(save_dir, "p3d_random_env_coverage_rate")
    plot_smooth_reward(paths, labels, save_path)

    print("Complete")


if __name__ == '__main__':
    # plot_static_env()
    plot_random_env()
