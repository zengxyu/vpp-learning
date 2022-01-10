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
import subprocess
import argparse
import logging
import numpy as np
from time import time

# Must be imported before large libs
from autoencoder.dataset.ae_dataset_with_features import make_data_loader_with_features
from autoencoder.network.vae_network import CompletionVAEShadowNet
from autoencoder.network.mink_unet34 import MinkUNetBase, MinkUNet34

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

if not os.path.exists("ModelNet40"):
    logging.info("Downloading the pruned ModelNet40 dataset...")
    subprocess.run(["sh", "./examples/download_modelnet40.sh"])

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d %H:%M:%S",
    handlers=[ch],
)

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=16)
parser.add_argument("--max_iter", type=int, default=30000)
parser.add_argument("--val_freq", type=int, default=100)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--stat_freq", type=int, default=50)
parser.add_argument("--weights", type=str, default="modelnet_features.pth")
parser.add_argument("--load_optimizer", type=str, default="true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--max_visualization", type=int, default=4)


###############################################################################
# End of utility functions
###############################################################################

def train(dataloader, device, config):
    net = MinkUNet34(4, 4).to(device)

    logging.info(net)

    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    weight = torch.Tensor(np.array([1 / 840, 1 / 15, 1 / 1, 1 / 1250]))
    crit = nn.BCEWithLogitsLoss(pos_weight=weight)

    net.train()
    train_iter = iter(dataloader)
    logging.info(f"LR: {scheduler.get_lr()}")

    for i in range(config.max_iter):

        s = time()
        data_dict = train_iter.next()
        d = time() - s

        optimizer.zero_grad()

        sin = ME.SparseTensor(
            features=data_dict["crop_feats"],
            coordinates=ME.utils.batched_coordinates(data_dict["tensor_batch_crop_coordinates"]),
            device=device,
        )

        # Generate target sparse tensor
        cm = sin.coordinate_manager
        target_key, _ = cm.insert_and_map(
            coordinates=ME.utils.batched_coordinates(data_dict["tensor_batch_truth_coordinates"]).to(device),
            string_id="target",
        )
        # Generate from a dense tensor
        # sinput = in_field.sparse()
        # Output sparse tensor
        code, sout = net(sin)
        # get the prediction on the input tensor field
        # out_field = soutput.slice(in_field)
        out_feature = sout.F
        target_feature = data_dict["crop_feats"]
        loss = crit(out_feature, target_feature.to(device))


        loss.backward()
        optimizer.step()
        t = time() - s

        # if i % config.stat_freq == 0:
        logging.info(
            f"Iter: {i}, Loss: {loss.item():.3e}, Data Loading Time: {d:.3e}, Tot Time: {t:.3e}"
        )

        if i % config.val_freq == 0 and i > 0:
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "curr_iter": i,
                },
                config.weights,
            )

            scheduler.step()
            logging.info(f"LR: {scheduler.get_lr()}")

            net.train()


if __name__ == "__main__":
    config = parser.parse_args()
    logging.info(config)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # path_to_data = "/media/zeng/Data/dataset/ModelNet40"
    # paths_to_data = ["/media/zeng/Data/dataset/Pheno4D/*/*.pcd"]
    # paths_to_data = ["/media/zeng/Data/dataset/Pheno4D/*/*.pcd",
    #                  "/media/zeng/Data/dataset/ModelNet40/chair/train/*.off"]

    paths_to_data = ["/home/zeng/catkin_ws/data/data_cvx/*.cvx"]
    # paths_to_data = ["/home/zeng/catkin_ws/data/data_cvx_test2/*.cvx"]

    dataloader = make_data_loader_with_features(
        paths_to_data,
        "train",
        augment_data=True,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        repeat=True,
        config=config,
        crop=True
    )

    train(dataloader, device, config)
