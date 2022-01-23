#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 1/22/22 5:00 PM 
    @Description    :
        
===========================================
"""
import os
from typing import Dict

import numpy as np
from pfrl import explorers

from configs.config import read_yaml
from utilities.util import get_project_path


def get_explorer_by_name(name, n_actions=None):
    config = read_yaml(config_dir=os.path.join(get_project_path(), "configs"), config_name="explorers.yaml")
    if name == "Greedy":
        return explorers.Greedy()
    elif name == "LinearDecayEpsilonGreedy":
        return explorers.LinearDecayEpsilonGreedy(
            start_epsilon=config[name]["start_epsilon"],
            end_epsilon=config[name]["end_epsilon"],
            decay_steps=config[name]["decay_steps"],
            random_action_func=lambda: np.random.randint(n_actions),
        )
    raise ValueError("Cannot find explorer by name {}".format(name))
