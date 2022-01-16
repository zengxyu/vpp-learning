import numpy as np

from autoencoder.dataset.data_reader import PointCloud

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

M = np.array(
    [
        [0.80656762, -0.5868724, -0.07091862],
        [0.3770505, 0.418344, 0.82632997],
        [-0.45528188, -0.6932309, 0.55870326],
    ]
)

# SCANNET_COLOR_MAP = {
#     0: (0., 0., 0.),
#     1: (174., 199., 232.),
#     2: (152., 223., 138.),
#     3: (31., 119., 180.)
# }

SCANNET_COLOR_MAP = {
    0: (255., 0., 0.),  # red
    1: (0., 255., 0.),  # green
    2: (0., 0., 255.),  # blue
    3: (31., 119., 180.)
}


def visualize_truth(data_dict, batch_ind, resolution, offset):
    point = data_dict["tensor_batch_truth_coordinates"][batch_ind]
    label = data_dict["tensor_batch_truth_feats"][batch_ind].numpy().squeeze()
    opcd = PointCloud(point, np.array([SCANNET_COLOR_MAP[l.argmax()] for l in label]))
    opcd.translate([offset * resolution, 0, 0])
    return opcd


def visualize_unfree_cells(data_dict, batch_ind, resolution, offset):
    point = data_dict["tensor_batch_truth_coordinates"][batch_ind]
    label = data_dict["tensor_batch_truth_feats"][batch_ind].numpy().squeeze()
    point = point[label.argmax(1) != 0]
    label = label[label.argmax(1) != 0]

    opcd = PointCloud(point, np.array([SCANNET_COLOR_MAP[l.argmax()] for l in label]))
    opcd.translate([offset * resolution, 0, 0])
    return opcd


def visualize_input(data_dict, batch_ind, resolution, offset):
    point = data_dict["tensor_batch_crop_coordinates"][batch_ind]
    label = data_dict["tensor_batch_crop_feats"][batch_ind].numpy().squeeze()
    opcd = PointCloud(point, np.array([SCANNET_COLOR_MAP[l.argmax()] for l in label]))
    opcd.translate([offset * resolution, 0, 0])
    return opcd


def visualize_prediction(point, feats, resolution, offset):
    point = point.detach().numpy()
    feats = feats.detach().numpy()
    pcd = PointCloud(point, np.array([SCANNET_COLOR_MAP[l.argmax()] for l in feats]))
    pcd.translate([offset * resolution, 0, 0])
    return pcd
