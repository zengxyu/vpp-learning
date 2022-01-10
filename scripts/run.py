import argparse
import logging
import os.path

import torch

from scripts.ae_config import get_parse_args
from scripts.autoencoder.utility import setup_logger
from scripts.autoencoder.learner import AELearner
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
