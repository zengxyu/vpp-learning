# File based on reconstruction.py example of MinkowskiEngine
# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import sys
# import subprocess
# import argparse
import logging
# import glob
import numpy as np
# from time import time
import urllib

# Must be imported before large libs
try:
    import open3d as o3d
except ImportError:
    raise ImportError(
        "Please install open3d and scipy with `pip install open3d scipy`."
    )

import torch
import torch.nn as nn
import torch.utils.data
# import torch.optim as optim
# from torch.utils.data.sampler import Sampler

import MinkowskiEngine as ME
from vpp_env_client import EnvironmentClient


class GenerativeNet(nn.Module):

    CHANNELS = [1024, 512, 256, 128, 64, 32, 16]

    def __init__(self, resolution, in_nchannel=512):
        nn.Module.__init__(self)

        self.resolution = resolution

        # Input sparse tensor must have tensor stride 128.
        ch = self.CHANNELS

        # Block 1
        self.block1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                in_nchannel, ch[0], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[0], ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[0], ch[1], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
        )

        self.block1_cls = ME.MinkowskiConvolution(
            ch[1], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 2
        self.block2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[1], ch[2], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[2], ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
        )

        self.block2_cls = ME.MinkowskiConvolution(
            ch[2], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 3
        self.block3 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[2], ch[3], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[3], ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
        )

        self.block3_cls = ME.MinkowskiConvolution(
            ch[3], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 4
        self.block4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[3], ch[4], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
        )

        self.block4_cls = ME.MinkowskiConvolution(
            ch[4], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 5
        self.block5 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[4], ch[5], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
        )

        self.block5_cls = ME.MinkowskiConvolution(
            ch[5], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 6
        self.block6 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[5], ch[6], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[6], ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
        )

        self.block6_cls = ME.MinkowskiConvolution(
            ch[6], 1, kernel_size=1, bias=True, dimension=3
        )

        # pruning
        self.pruning = ME.MinkowskiPruning()

    @torch.no_grad()
    def get_target(self, out, target_key, kernel_size=1):
        target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        cm = out.coordinate_manager
        strided_target_key = cm.stride(
            target_key,
            out.tensor_stride[0],
        )
        kernel_map = cm.kernel_map(
            out.coordinate_map_key,
            strided_target_key,
            kernel_size=kernel_size,
            region_type=1,
        )
        for k, curr_in in kernel_map.items():
            target[curr_in[0].long()] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, z, target_key):
        out_cls, targets = [], []

        # Block1
        out1 = self.block1(z)
        out1_cls = self.block1_cls(out1)
        target = self.get_target(out1, target_key)
        targets.append(target)
        out_cls.append(out1_cls)
        keep1 = (out1_cls.F > 0).squeeze()

        # If training, force target shape generation, use net.eval() to disable
        if self.training:
            keep1 += target

        # Remove voxels 32
        out1 = self.pruning(out1, keep1)

        # Block 2
        out2 = self.block2(out1)
        out2_cls = self.block2_cls(out2)
        target = self.get_target(out2, target_key)
        targets.append(target)
        out_cls.append(out2_cls)
        keep2 = (out2_cls.F > 0).squeeze()

        if self.training:
            keep2 += target

        # Remove voxels 16
        out2 = self.pruning(out2, keep2)

        # Block 3
        out3 = self.block3(out2)
        out3_cls = self.block3_cls(out3)
        target = self.get_target(out3, target_key)
        targets.append(target)
        out_cls.append(out3_cls)
        keep3 = (out3_cls.F > 0).squeeze()

        if self.training:
            keep3 += target

        # Remove voxels 8
        out3 = self.pruning(out3, keep3)

        # Block 4
        out4 = self.block4(out3)
        out4_cls = self.block4_cls(out4)
        target = self.get_target(out4, target_key)
        targets.append(target)
        out_cls.append(out4_cls)
        keep4 = (out4_cls.F > 0).squeeze()

        if self.training:
            keep4 += target

        # Remove voxels 4
        out4 = self.pruning(out4, keep4)

        # Block 5
        out5 = self.block5(out4)
        out5_cls = self.block5_cls(out5)
        target = self.get_target(out5, target_key)
        targets.append(target)
        out_cls.append(out5_cls)
        keep5 = (out5_cls.F > 0).squeeze()

        if self.training:
            keep5 += target

        # Remove voxels 2
        out5 = self.pruning(out5, keep5)

        # Block 5
        out6 = self.block6(out5)
        out6_cls = self.block6_cls(out6)
        target = self.get_target(out6, target_key)
        targets.append(target)
        out_cls.append(out6_cls)
        keep6 = (out6_cls.F > 0).squeeze()

        # Last layer does not require keep
        # if self.training:
        #   keep6 += target

        # Remove voxels 1
        out6 = self.pruning(out6, keep6)

        return out_cls, targets, out6


def PointCloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def pointcloudToNetInput(xyz, resolution, feats):
    # Use color or other features if available
    # feats = np.ones((len(xyz), 1))

    # Get coords
    # xyz = xyz * resolution
    # features = np.ones((len(xyz), 1))  # torch.ones((len(xyz), 1))
    # coords, feats = ME.utils.sparse_quantize(xyz, features)

    # return (coords, feats)

    print('Input', xyz.shape, feats.shape)

    xyz = xyz * resolution
    coords, features, inds = ME.utils.sparse_quantize(xyz, features=np.expand_dims(feats, axis=1), return_index=True)

    print('Output', coords.shape, inds.shape, features.shape)

    return (coords, xyz[inds], features)


def collate_pointcloud_fn(list_data):
    coords, feats, labels = list(zip(*list_data))

    # Concatenate all lists
    return {
        "coords": coords,
        "xyzs": [torch.from_numpy(feat).float() for feat in feats],
        "labels": labels
    }


def visualize(net, dataloader, device, in_nchannel, batch_size, max_visualization=4):
    crit = nn.BCEWithLogitsLoss()
    n_vis = 0

    for data_dict in dataloader:

        init_coords = torch.zeros((batch_size, 4), dtype=torch.int)
        init_coords[:, 0] = torch.arange(batch_size)

        in_feat = torch.zeros((batch_size, in_nchannel))
        in_feat[torch.arange(batch_size), data_dict["labels"]] = 1

        sin = ME.SparseTensor(
            features=in_feat,
            coordinates=init_coords,
            tensor_stride=net.resolution,
            device=device,
        )

        # Generate target sparse tensor
        cm = sin.coordinate_manager
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(data_dict["xyzs"]).to(device),
            string_id="target",
        )

        # Generate from a dense tensor
        out_cls, targets, sout = net(sin, target_key)
        num_layers, loss = len(out_cls), 0
        for out_cl, target in zip(out_cls, targets):
            loss += (
                crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device))
                / num_layers
            )

        batch_coords, batch_feats = sout.decomposed_coordinates_and_features
        for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
            pcd = PointCloud(coords.cpu())
            pcd.estimate_normals()
            pcd.translate([0.6 * net.resolution, 0, 0])
            # pcd.rotate(M)
            opcd = PointCloud(data_dict["xyzs"][b])
            opcd.translate([-0.6 * net.resolution, 0, 0])
            opcd.estimate_normals()
            # opcd.rotate(M)
            o3d.visualization.draw_geometries([pcd, opcd])

            n_vis += 1
            if n_vis > max_visualization:
                return


def visualize_one(net, xyz, device):  # not working
    coords, feats = pointcloudToNetInput(xyz, net.resolution)

    sin = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=net.resolution, device=device)

    # Generate target sparse tensor
    cm = sin.coordinate_manager
    target_key, _ = cm.insert_and_map(coords.to(device), string_id="target")

    # manager = ME.CoordinateManager(D=2)
    # key, _ = manager.insert_and_map(coords.to(device), [2])

    # Generate from a dense tensor
    out_cls, targets, sout = net(sin, target_key)

    coords_recovered, feats_recovered = sout.decomposed_coordinates_and_features

    pcd = PointCloud(coords_recovered.cpu())
    pcd.estimate_normals()
    pcd.translate([3.0 * net.resolution, 0, 0])
    opcd = PointCloud(coords)
    opcd.estimate_normals()
    opcd.translate([-3.0 * net.resolution, 0, 0])
    o3d.visualization.draw_geometries([pcd, opcd])


def main(args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    resolution = 128
    in_nchannel = 4
    weights = "modelnet_reconstruction.pth"
    batch_size = 4

    net = GenerativeNet(resolution, in_nchannel=in_nchannel)
    net.to(device)

    logging.info(net)

    if not os.path.exists(weights):
        logging.info(f"Downloaing pretrained weights. This might take a while...")
        urllib.request.urlretrieve("https://bit.ly/36d9m1n", filename=weights)

    logging.info(f"Loading weights from {weights}")
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    state_dict["block1.0.kernel"] = state_dict["block1.0.kernel"][:, 0:in_nchannel, :]
    net.load_state_dict(state_dict)
    net.eval()

    client = EnvironmentClient(handle_simulation=False)
    points, labels, robotPose, robotJoints, reward = client.sendReset(map_type='pointcloud')

    dataset = []
    for i in range(in_nchannel):
        dataset.append(pointcloudToNetInput(points, resolution, labels))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_pointcloud_fn)

    print(dataloader)

    visualize(net, dataloader, device, in_nchannel, batch_size)

    # visualize_one(net, points, device)


if __name__ == '__main__':
    main(sys.argv)
