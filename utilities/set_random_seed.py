#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/9/22 3:48 PM 
    @Description    :
        
===========================================
"""
import pfrl
import os

import torch
import random
import gym
import numpy as np


def set_random_seeds(random_seed: int):
    """
    Setup all possible random seeds so results can be reproduced
    """
    pfrl.utils.random_seed.set_random_seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_seed)
    # tf.set_random_seed(random_seed) # if you use tensorflow
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)
    if hasattr(gym.spaces, "prng"):
        gym.spaces.prng.seed(random_seed)
