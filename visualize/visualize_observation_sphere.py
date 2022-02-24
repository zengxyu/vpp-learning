#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/24/22 3:44 PM 
    @Description    :
        
===========================================
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation

vec_apply = np.vectorize(Rotation.apply, otypes=[np.ndarray], excluded=['vectors', 'inverse'])


def visualize_observation_sphere():
    fig = plt.figure()
    ax = Axes3D(fig)

    rots = compute_vecs()
    rots = np.reshape(np.array(rots), (-1, 3))
    rot_num = np.linalg.norm(rots[0])
    depth = 5
    frac = rot_num / depth
    points = []
    for direction in rots:
        for i in range(depth):
            points.append(i * frac * direction)

    points = np.array(points) * 5
    x = np.array(points[:, 0], np.float)
    y = np.array(points[:, 1], np.float)
    z = np.array(points[:, 2], np.float)
    ax.scatter(x, y, z)
    # ax.stem(x, y, z)
    plt.show()


def compute_vecs():
    robot_rot = Rotation.from_quat([0, 0, 0, 1])
    axes = robot_rot.as_matrix().transpose()
    rh = np.radians(np.linspace(-180, 180, 36))
    rv = np.radians(np.linspace(0, 180, 18))
    rots_x = Rotation.from_rotvec(np.outer(rh, axes[2]))
    rots_y = Rotation.from_rotvec(np.outer(rv, axes[1]))

    rots = vec_apply(np.outer(rots_x, rots_y), vectors=axes[0])
    rots = np.reshape(np.array(rots), (-1,))

    rots2 = []
    for rot in rots:
        rots2.append([rot[0], rot[1], rot[2]])
    rots2 = np.array(rots2)
    return rots2

if __name__ == '__main__':
    visualize_observation_sphere()
