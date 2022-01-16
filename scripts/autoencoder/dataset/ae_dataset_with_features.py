import os
import random
import sys
import glob
import numpy as np

from torch.utils.data.sampler import Sampler
import torch
import torch.utils.data
import MinkowskiEngine as ME
import capnp

from autoencoder.dataset.data_reader import InfSampler, normalize

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', "capnp"))
import pointcloud_capnp
import voxelgrid_capnp


class AEDatasetWithFeatures(torch.utils.data.Dataset):
    # TODO 改这个地方
    def __init__(self, config=None):
        self.phase = config.phase
        self.cache = {}
        self.resolution = config.resolution
        # load from path
        fnames = []
        for paths_to_data in config.paths_to_data:
            fnames.extend(glob.glob(paths_to_data))
        self.files = fnames
        # label0count = 0
        # label1count = 0
        # label2count = 0
        # label3count = 0
        # # loading into cache
        for file_path in self.files:
            file = open(file_path, 'rb')
            if file_path.endswith("cpc"):
                pointcloud = pointcloud_capnp.Pointcloud.read(file, traversal_limit_in_words=2 ** 63)
                points = np.reshape(np.array(pointcloud.points), (-1, 3))
                labels = np.array(pointcloud.labels)
                points = points[np.where(labels != 0)]
                labels_roi_one_hot = to_one_hot(labels - 1, class_num=2)

            else:
                voxelgrid = voxelgrid_capnp.Voxelgrid.read(file, traversal_limit_in_words=2 ** 63)
                shape = np.array(voxelgrid.shape)
                points = []
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        for k in range(shape[2]):
                            points.append([i, j, k])
                points = np.array(points)
                labels = np.array(voxelgrid.labels)
                # label0count += np.sum(labels == 0)
                # label1count += np.sum(labels == 1)
                # label2count += np.sum(labels == 2)
                # label3count += np.sum(labels == 3)

                labels_roi_one_hot = to_one_hot(labels, class_num=4)
            # TODO normalize
            # points = np.reshape(np.array(pointcloud.points), (-1, 3))
            # labels = np.array(pointcloud.labels).tolist()

            # points_roi = points[np.where(labels == 2)]
            # labels_roi = labels[np.where(labels == 2)]
            # points = normalize(points)

            if len(points) >= 100:
                # points_roi = normalize(points_roi)
                self.cache[len(self.cache)] = [points, labels_roi_one_hot]
                print("Add file:{} into cache!".format(file))
            else:
                print("File:{} too small!".format(file))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        points, labels = self.cache[idx]
        points = normalize(points)
        coords, _, feats = quantize_coordinates_with_feats(points, feats=labels,
                                                           resolution=self.resolution)
        # TODO rotate the pointcloud
        # coords_crop, feats_crop, coords_truth, feats_truth = random_crop(coords, feats, self.resolution,
        #                                                                  partial_rate=0.5)

        # return coords_crop, feats_crop, coords_truth, feats_truth, idx
        # print("coords.shape:", coords.shape)
        # print("feats.shape:", feats.shape)
        return coords, feats, coords, feats, idx


def load_one_sample(voxelgrid, config):
    shape = np.array(voxelgrid.shape)
    points = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                points.append([i, j, k])
    points = np.array(points)
    labels = np.array(voxelgrid.labels)
    labels = to_one_hot(labels, class_num=4)

    points = normalize(points)
    coords, _, feats = quantize_coordinates_with_feats(points, feats=labels,
                                                       resolution=config.resolution)
    coords = [coords]
    feats = [feats]
    data_batch_dict = construct_data_batch(coords, feats, coords, feats)
    return data_batch_dict


def quantize_coordinates_with_feats(xyz, feats, resolution):
    """

    Args:
        xyz:
        feats:
        resolution:

    Returns: quantized coordinates,
             original coordinates at inds of quantized coordinates,
             features at inds of quantized coordinates

    """
    # Use labels (free, occupied, ROI) as features
    if feats.ndim < 2:
        feats = np.expand_dims(feats, axis=1)
    # feats = np.ones((len(xyz), 1))

    # Get coords
    xyz = xyz * resolution
    quantized_coords, feats_at_inds, inds = ME.utils.sparse_quantize(xyz, features=feats, return_index=True)

    original_coords_at_inds = xyz[inds]

    return quantized_coords, original_coords_at_inds, feats_at_inds


def to_one_hot(indexes, class_num):
    one_hot_ = []
    for ind in indexes:
        one_hot_.append(np.eye(class_num, dtype=np.int8)[ind])
    return np.array(one_hot_)


def make_data_loader_with_features(config):
    dset = AEDatasetWithFeatures(config)
    print("dataset size:{}".format(len(dset)))

    args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "collate_fn": CollationAndTransformation(config.resolution),
        "pin_memory": False,
        "drop_last": False,
    }

    if config.repeat:
        args["sampler"] = InfSampler(dset, config.shuffle)
    else:
        args["shuffle"] = config.shuffle

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader


class CollationAndTransformation:
    def __init__(self, resolution):
        self.resolution = resolution

    def random_crop(self, data_list):
        # coords_list, ori_coords_list, feats_list = data_list
        # coords_list, ori_coords_list = data_list
        crop_coords_list = []
        crop_ori_coords_list = []
        crop_feats_list = []
        if len(data_list) == 3:
            for coords, ori_coords, feats in zip(*data_list):
                rand_idx = random.randint(0, int(self.resolution * 0.66))
                sel0 = coords[:, 0] > rand_idx
                max_range = self.resolution / 3 + rand_idx
                sel1 = coords[:, 0] < max_range
                sel = sel0 * sel1
                crop_coords_list.append(coords[sel])
                crop_ori_coords_list.append(ori_coords[sel])
                crop_feats_list.append(feats[sel])
            return crop_coords_list, crop_ori_coords_list, crop_feats_list
        else:
            for coords, ori_coords in zip(*data_list):
                rand_idx = random.randint(0, int(self.resolution * 0.66))
                sel0 = coords[:, 0] > rand_idx
                max_range = self.resolution / 3 + rand_idx
                sel1 = coords[:, 0] < max_range
                sel = sel0 * sel1
                crop_coords_list.append(coords[sel])
                crop_ori_coords_list.append(ori_coords[sel])
            return crop_coords_list, crop_ori_coords_list

    def __call__(self, list_data):
        coords_crop, feats_crop, coords_truth, feats_truth, indx = list(zip(*list_data))
        item = construct_data_batch(coords_crop, feats_crop, coords_truth, feats_truth)
        # else:
        #     coords, ori_coords, indx = list(zip(*list_data))
        #     item = construct_data_batch(coords, ori_coords)
        return item


def construct_data_batch(coords_crop, feats_crop, coords_truth, feats_truth):
    coords_crop_batch = ME.utils.batched_coordinates(coords_crop)
    feats_crop_batch = ME.utils.batched_coordinates(feats_crop)

    data_batch_dict = {
        # "batched_crop_coordinates": coords_crop_batch,
        "crop_feats": torch.cat([torch.Tensor(feats).float() for feats in feats_crop]),
        "tensor_batch_crop_coordinates": [coord.float() for coord in coords_crop],
        "tensor_batch_crop_feats": [torch.from_numpy(feats).float() for feats in feats_crop],
        "tensor_batch_truth_coordinates": [coord_truth for coord_truth in coords_truth],
        "tensor_batch_truth_feats": [torch.from_numpy(feats).float() for feats in feats_truth]
    }
    return data_batch_dict
