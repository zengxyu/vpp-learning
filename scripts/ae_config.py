import argparse
import logging
import os.path

import torch


def get_parse_args():
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
    parser.add_argument("--weights", type=str, default="model/modelnet_features.pth")
    parser.add_argument("--load_optimizer", type=str, default="true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--use_gpu", type=bool, default=False)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--repeat", type=bool, default=False)
    parser.add_argument("--phase", type=str, default="eval", help="choose from train or eval or inference")
    parser.add_argument("--dir_to_data", type=str, default="/home/zeng/catkin_ws/data/data_cvx_test2",
                        help="path to the data directory")
    # "/home/zeng/catkin_ws/data/data_cvx/*.cvx"
    # "/home/zeng/catkin_ws/data/data_cvx_test2/*.cvx"
    config = parser.parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = torch.device("cpu")

    # data dir
    config.paths_to_data = [config.dir_to_data + "/*.cvx"]
    config.weights = os.path.join(os.path.dirname(__file__), config.weights)
    model_dir = os.path.join(os.path.dirname(__file__), config.weights[:config.weights.rindex("/")])
    # create dir to save the model.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return config
