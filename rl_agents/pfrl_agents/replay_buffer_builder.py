#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 1/22/22 4:30 PM 
    @Description    :
        
===========================================
"""
import os.path

from pfrl import replay_buffers

from configs.config import read_yaml
from utilities.util import get_project_path


def get_replay_buffer_by_name(name):
    config = read_yaml(config_dir=os.path.join(get_project_path(), "configs"), config_name="replay_buffers.yaml")
    if name == "ReplayBuffer":
        return replay_buffers.ReplayBuffer(
            capacity=config.getint(name, "capacity"),
            num_steps=config.getint(name, "num_steps"),
        )
    elif name == "PrioritizedReplayBuffer":
        return replay_buffers.PrioritizedReplayBuffer(
            capacity=config[name]["capacity"],
            num_steps=config[name]["num_steps"],
            alpha=config[name]["alpha"],
            beta0=config[name]["beta0"],
            betasteps=config[name]["betasteps"],
            normalize_by_max=config[name]["normalize_by_max"],
        )
    else:
        raise NotImplementedError("Cannot find replay buffer by name : {}".format(name))
