#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/20/22 11:48 AM 
    @Description    :
        
===========================================
"""
import os
import pickle

import numpy as np


def save_observation_map(observation_map: np.array, step_count: int, parser_args):
    """
    save observation map
    """
    if parser_args.env_config["save_obs"]:
        to_dir = os.path.join(parser_args.out_result, "observation_map")
        if not os.path.exists(to_dir):
            os.makedirs(to_dir)
            print("create folder : {}".format(to_dir))
        to_path = os.path.join(to_dir, "obs_{}.obj".format(step_count))
        pickle.dump(observation_map, open(to_path, 'wb'))
        print("Save observation map to : {}".format(to_path))
