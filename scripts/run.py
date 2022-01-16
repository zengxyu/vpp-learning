import argparse
import logging
import os.path

import torch
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "autoencoder"))
sys.path.append(os.path.join(os.path.dirname(__file__), "autoencoder", "dataset"))

from ae_config import get_parse_args
from autoencoder.utility import setup_logger
from autoencoder.learner import AELearner
from vpp_env_client import EnvironmentClient

setup_logger()

if __name__ == "__main__":
    config = get_parse_args()
    logging.info(config)

    learner = AELearner(config)
    if config.phase == "train":
        learner.training()
    elif config.phase == "eval":
        learner.evaluation()
    else:
        client = EnvironmentClient(handle_simulation=False)
        voxelgrid, robotPose, robotJoints, reward = client.sendReset(map_type='voxelgrid')
        learner.inference(voxelgrid)
