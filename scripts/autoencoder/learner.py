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

import logging
import numpy as np
import time
# Must be imported before large libs
from dataset.ae_dataset_with_features import make_data_loader_with_features, load_one_sample
from network.mink_unet34 import MinkUNet34
from visualize import visualize_prediction, visualize_input, visualize_truth

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

import MinkowskiEngine as ME

M = np.array(
    [
        [0.80656762, -0.5868724, -0.07091862],
        [0.3770505, 0.418344, 0.82632997],
        [-0.45528188, -0.6932309, 0.55870326],
    ]
)

assert (
        int(o3d.__version__.split(".")[1]) >= 8
), f"Requires open3d version >= 0.8, the current version is {o3d.__version__}"


class AELearner:
    def __init__(self, config):
        self.config = config
        self.net = MinkUNet34(4, 4).to(config.device)
        logging.info(self.net)

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.config.lr, momentum=self.config.momentum,
                                   weight_decay=self.config.weight_decay)

        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)

        weight = torch.Tensor(np.array([1 / 840, 1 / 15, 1 / 1, 1 / 1250]))
        self.crit = nn.BCEWithLogitsLoss(pos_weight=weight)

        if self.config.phase != "train":
            self.load_model()

    def training(self):
        self.net.train()

        dataloader = make_data_loader_with_features(self.config)

        train_iter = iter(dataloader)
        logging.info(f"LR: {self.scheduler.get_lr()}")

        for i in range(self.config.max_iter):

            s = time.time()
            data_dict = train_iter.next()
            d = time.time() - s

            self.optimizer.zero_grad()
            sinput, target_key = self.generate_input_and_target(data_dict)

            # get the prediction on the input tensor field
            code, sout = self.net(sinput)

            # compute loss
            out_feature = sout.F
            target_feature = data_dict["crop_feats"]
            loss = self.crit(out_feature, target_feature.to(self.config.device))

            loss.backward()
            self.optimizer.step()
            t = time.time() - s

            if i % self.config.stat_freq == 0:
                logging.info(
                    f"Iter: {i}, Loss: {loss.item():.3e}, Data Loading Time: {d:.3e}, Tot Time: {t:.3e}"
                )

            if i % self.config.val_freq == 0 and i > 0:
                self.save_model(i)

                self.scheduler.step()

                logging.info(f"LR: {self.scheduler.get_lr()}")

                self.net.train()

    def evaluation(self):
        logging.info(self.net)

        self.net.eval()

        dataloader = make_data_loader_with_features(self.config)

        for data_dict in dataloader:
            start_time = time.time()
            sinput, target_key = self.generate_input_and_target(data_dict)
            code, sout = self.net(sinput)
            print("inference takes time:{}".format(time.time() - start_time))
            # batch_code_coord, batch_code_feats = code.decomposed_coordinates_and_features
            self.visualize(data_dict, sout)

    def inference(self, voxelgrid):
        s_time = time.time()

        self.net.eval()

        data_batch_dict = load_one_sample(voxelgrid, self.config)
        sinput, target_key = self.generate_input_and_target(data_batch_dict)
        code, sout = self.net(sinput)

        coordinates = code.coordinates.cpu().detach().numpy()[:, 0]
        index = coordinates.argsort()
        features = code.features.cpu().detach().numpy()[index]
        # self.visualize(data_batch_dict, sout)
        print("inference takes {} s\n".format(time.time() - s_time))
        return features

    def visualize(self, data_dict, sout):
        batch_coords, batch_feats = sout.decomposed_coordinates_and_features

        for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
            pcd_input = visualize_input(data_dict, b, self.config.resolution, offset=0)
            pcd_truth = visualize_truth(data_dict, b, self.config.resolution, offset=2)
            # pcd_truth2 = visualize_truth2(data_dict, b)
            pcd = visualize_prediction(coords, feats, self.config.resolution, offset=4)
            o3d.visualization.draw_geometries([pcd_input, pcd_truth, pcd])

    def load_model(self):
        logging.info(f"Loading weights from {self.config.weights}")
        checkpoint = torch.load(self.config.weights)
        self.net.load_state_dict(checkpoint["state_dict"])

    def generate_input_and_target(self, data_dict):
        """
        # Generate from a dense tensor
        # Output sparse tensor
        Args:
            data_dict:

        Returns:

        """
        sinput = ME.SparseTensor(
            features=data_dict["crop_feats"],
            coordinates=ME.utils.batched_coordinates(data_dict["tensor_batch_crop_coordinates"]),
            device=self.config.device,
        )

        # Generate target sparse tensor
        cm = sinput.coordinate_manager
        target_key, _ = cm.insert_and_map(
            coordinates=ME.utils.batched_coordinates(data_dict["tensor_batch_truth_coordinates"]).to(
                self.config.device),
            string_id="target",
        )
        return sinput, target_key

    def save_model(self, i):
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "curr_iter": i,
            },
            self.config.weights,
        )
