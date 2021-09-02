import pickle

import numpy as np
# 方法一，利用关键字
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax1 = plt.axes(projection='3d')


def plot_line(x, y, z):
    # 定义坐标轴

    # ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图

    ax1.plot3D(x, y, z, 'gray')  # 绘制空间曲线
    plt.show()


def plot_point(xd, yd, zd):
    ax1.scatter3D(xd, yd, zd, cmap='Blues')  # 绘制散点图
    plt.show()


def read_path_data(file_path):
    arr = pickle.load(open(file_path, "rb"))
    path = arr[3]
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


if __name__ == '__main__':
    gxs, gys, gzs, free_xs, free_ys, free_zs = read_global_map()
    ax1.scatter3D(gxs, gys, gzs, cmap='Blues')  # 绘制散点图

    file_path = "/home/zeng/workspace/vpp-learning/output/predict_p3d_static_pose_lstm/path_bak2.obj"
    xs, ys, zs = read_path_data(file_path)
    xs, ys, zs = xs[:300], ys[:300], zs[:300]
    ax1.scatter3D(xs, ys, zs, cmap='Blacks')  # 绘制散点图
    ax1.plot3D(xs, ys, zs, 'gray')  # 绘制散点图
    # ax1.scatter3D(free_xs, free_ys, free_zs)  # 绘制散点图

    plt.show()