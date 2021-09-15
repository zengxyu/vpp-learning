import os
import pickle

import matplotlib
import numpy as np
# 方法一，利用关键字
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D

from utilities.util import get_project_path

font_size = 12

font1 = {'size': font_size,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': font_size + 4,
         }


def read_path_data(file_path):
    path = pickle.load(open(file_path, "rb"))
    xs = []
    ys = []
    zs = []
    for robot_state in path:
        x, y, z, _, _, _, _ = robot_state
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return xs, ys, zs


def read_global_map(path):
    global_map = pickle.load(open(path, "rb"))
    max_v = np.max(global_map)
    print("max value:{}".format(max_v))
    w, h, d = global_map.shape
    fruit_xs, fruit_ys, fruit_zs = [], [], []
    free_xs, free_ys, free_zs = [], [], []

    for x in range(w):
        for y in range(h):
            for z in range(d):
                if global_map[x, y, z] == 2:
                    fruit_xs.append(x)
                    fruit_ys.append(y)
                    fruit_zs.append(z)
                if global_map[x, y, z] == 1:
                    free_xs.append(x)
                    free_ys.append(y)
                    free_zs.append(z)
    return fruit_xs, fruit_ys, fruit_zs, free_xs, free_ys, free_zs


def plot_global_map(global_map_path):
    gxs, gys, gzs, free_xs, free_ys, free_zs = read_global_map(global_map_path)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.view_init(10, -75)
    p2 = ax1.scatter(gzs, gys, gxs, c='#99ff33', marker='o')  # plot tree points
    ax1.set_xlim(0, 256)
    ax1.set_ylim(0, 256)
    ax1.set_zlim(0, 256)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    plt.show()


def plot1(global_map_path, path_path, out_dir_path):
    xs, ys, zs = read_path_data(path_path)
    gxs, gys, gzs, free_xs, free_ys, free_zs = read_global_map(global_map_path)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.view_init(10, -75)

    p1 = ax1.scatter(zs, ys, xs, s=20, c='#ff3333', marker='x')  # plot path points
    line1 = ax1.plot(zs, ys, xs, c='#111111')  # plot path line

    p2 = ax1.scatter(gzs, gys, gxs, c='#99ff33', marker='o')  # plot tree points

    plt.legend(handles=[p1, p2, line1], labels=['viewpoint location', 'leaves', 'path'], loc='best', prop=font1)
    # ax1.legend(handles=[p1, p2, line1], labels=['viewpoint location', 'leaves', 'path'], loc='best', fontsize=font_size)
    # ax1.scatter3D(free_xs, free_ys, free_zs)  # 绘制散点图
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    ax1.set_xlim(0, 256)
    ax1.set_ylim(0, 256)
    ax1.set_zlim(0, 256)

    ax1.set_xlabel("x", fontsize=font_size + 2)
    ax1.set_ylabel("y", fontsize=font_size + 2)
    ax1.set_zlabel("z", fontsize=font_size + 2)

    ax1.view_init(10, -45)
    out_path = os.path.join(out_dir_path, "path_random_env_plot.png")
    plt.savefig(out_path)
    # plt.show()
    plt.show()


if __name__ == '__main__':
    out_dir = os.path.join(get_project_path(), "plot", "p3d_evaluation_plot")
    in_parent_dir = "/media/zeng/Workspace/results_paper/out_p3d_final/evaluation"
    in_dir = os.path.join(in_parent_dir, "out_p3d_random_step_len_10_36_action_predict_model_550_save_env_and_path")
    loss, rewards = pickle.load(open(os.path.join(in_dir, "loss_reward", "loss_reward.obj"), "rb"))
    print(rewards)
    path_path = os.path.join(in_dir, "path_5.obj")
    global_map_path = os.path.join(in_dir, "global_map_5.obj")

    plot1(global_map_path, path_path, out_dir)
