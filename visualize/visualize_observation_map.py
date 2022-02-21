import pickle
import argparse
import cv2
import torch
import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation

from utilities.util import get_project_path

vec_apply = np.vectorize(Rotation.apply, otypes=[np.ndarray], excluded=['vectors', 'inverse'])


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_to", type=str, default="test_folder")
    parser.add_argument("--in_path", type=str, default=None)

    parser_args = parser.parse_args()
    return parser_args


def save_image(name, image):
    image_path = os.path.join(save_path, name)
    cv2.imwrite(image_path, image * 256)


def minmaxscaler(data):
    min = torch.min(data)
    max = torch.max(data)
    return (data - min) / (max - min)


def con_frame(frame):
    con_frames = frame[0]
    for i in range(1, 15, 1):
        br = np.ones((frame[i].shape[0], 2))
        con_frames = np.hstack([con_frames, br, frame[i]])

    return con_frames


def con_frame2(frame, title):
    "将一个observation map合成彩色图像"
    results = None
    for i in range(5):
        a = [frame[0 + i]]
        a.append(frame[5 + i])
        a.append(frame[10 + i])
        a = np.array(a)
        a = a.transpose((1, 2, 0))
        if results is None:
            results = [a]
        else:
            results.append(a)
        save_image(title + "_{}.png".format(i), a)
    results = np.hstack(results)
    return results


def display_image(image, title):
    scale = 10
    image = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale))
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_observation_map(path):
    observation_map = pickle.load(open(path, "rb"))
    return observation_map


def explode(data):
    size = np.array(data.shape) * 2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e


def get_observation_map():
    dir_path = os.path.join(get_project_path(), "output", "test_folder", "result_log")
    obs_rois = {}
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        observation_map = load_observation_map(file_path)
        observation_map = np.reshape(observation_map, (4, 5, 36, 18))
        observation_map = np.transpose(observation_map, (0, 2, 1, 3))
        observation_map = np.reshape(observation_map, (4, 36, 90))
        obs_roi = observation_map[1]
        obs_rois[file_name] = obs_roi
    pass


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
    # get_observation_map()
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

    # # build up the numpy logo
    # n_voxels = np.zeros((4, 7, 4), dtype=bool)
    # n_voxels[0, 0, :] = True
    # n_voxels[-1, 0, :] = True
    # n_voxels[1, 0, 2] = True
    # n_voxels[2, 0, 1] = True
    # facecolors = np.where(n_voxels, '#FFD65D00', '#7A88CCFF')
    # edgecolors = np.where(n_voxels, '#BFAB6E', '#7D84A6')
    # filled = np.ones(n_voxels.shape)
    #
    # # upscale the above voxel image, leaving gaps
    # filled_2 = explode(filled)
    # fcolors_2 = explode(facecolors)
    # ecolors_2 = explode(edgecolors)
    #
    # # Shrink the gaps
    # x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    # x[0::2, :, :] += 0.05
    # y[:, 0::2, :] += 0.05
    # z[:, :, 0::2] += 0.05
    # x[1::2, :, :] += 0.95
    # y[:, 1::2, :] += 0.95
    # z[:, :, 1::2] += 0.95
    #
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
    # plt.savefig(os.path.join(get_project_path(), "output", "matplot.png"))
    # plt.show()
