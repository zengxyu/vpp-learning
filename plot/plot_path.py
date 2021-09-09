import pickle

import matplotlib
import numpy as np
# 方法一，利用关键字
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D


# def plot_line(x, y, z):
#     # 定义坐标轴
#
#     # ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图
#
#     ax1.plot3D(x, y, z, 'gray')  # 绘制空间曲线
#     plt.show()
#
#
# def plot_point(xd, yd, zd):
#     ax1.scatter3D(xd, yd, zd, cmap='Blues')  # 绘制散点图
#     plt.show()


def read_path_data(file_path):
    arr = pickle.load(open(file_path, "rb"))
    path = arr[0]
    xs = []
    ys = []
    zs = []
    for robot_state in path:
        x, y, z, _, _, _, _ = robot_state
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return xs, ys, zs


def read_global_map():
    path = "/home/zeng/workspace/vpp-learning/global_map.obj"
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


def plot1():
    file_path = "/home/zeng/workspace/vpp-learning/output/predict_p3d_static_pose_lstm/path.obj"
    xs, ys, zs = read_path_data(file_path)
    gxs, gys, gzs, free_xs, free_ys, free_zs = read_global_map()

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.view_init(10, -75)

    p1 = ax1.scatter(zs, ys, xs, s=20, c='#ff3333', marker='x')  # 绘制散点图
    line1 = ax1.plot(zs, ys, xs, c='#111111')  # 绘制散点图

    p2 = ax1.scatter(gzs, gys, gxs, c='#99ff33', marker='o')  # 绘制散点图

    plt.legend(handles=[p1, p2, line1], labels=['viewpoint location', 'leaves', 'path'], loc='best')
    # ax1.scatter3D(free_xs, free_ys, free_zs)  # 绘制散点图
    ax1.set_xlim(0, 256)
    ax1.set_ylim(0, 256)
    ax1.set_zlim(0, 256)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    plt.show()


def plot2():
    # ax1 = plt.axes(projection='3d')
    file_path = "/home/zeng/workspace/vpp-learning/output/predict_p3d_static_pose_lstm/path_bak2.obj"
    xs, ys, zs = read_path_data(file_path)
    xs, ys, zs = xs[:300], ys[:300], zs[:300]
    gxs, gys, gzs, free_xs, free_ys, free_zs = read_global_map()

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    all_x = np.append(xs, gxs)
    all_y = np.append(ys, gys)
    all_z = np.append(zs, gzs)
    all_c = np.append(np.zeros((len(xs),), np.ones((len(gxs)))))
    ax1.scatter(all_x, all_y, all_z, s=20, c=all_c, marker='x', cmap=plt.get_cmap("Greens"))  # 绘制散点图
    ax1.plot(zs, ys, xs, 'gray')  # 绘制散点图
    #
    # # cg = np.lin
    # gvs = np.ones(shape=(len(gxs),)) * 1
    # cg = np.linspace(0.4, 0.5, len(gzs))
    # cg[0] = 0
    # cg[-1] = 1
    # ax1.scatter(gzs, gys, gxs, c=cg, marker='o', cmap='Greens')  # 绘制散点图

    # ax1.scatter3D(free_xs, free_ys, free_zs)  # 绘制散点图
    ax1.set_xlim(0, 256)
    ax1.set_ylim(0, 256)
    ax1.set_zlim(0, 256)
    ax1.view_init(10, -70)
    # plt.savefig()
    plt.show()


if __name__ == '__main__':
    plot1()
