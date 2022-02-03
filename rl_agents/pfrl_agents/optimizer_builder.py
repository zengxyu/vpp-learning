#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 1/22/22 5:13 PM 
    @Description    :
        
===========================================
"""
import os

import torch
from pfrl import optimizers

from config import read_yaml, get_configs_dir
from utilities.util import get_project_path


def get_optimizer_by_name(name, network):
    config = read_yaml(config_dir=get_configs_dir(), config_name="optimizers.yaml")
    if name == "RMSpropEpsInsideSqrt":
        return optimizers.RMSpropEpsInsideSqrt(
            network.parameters(),
            lr=config[name]["lr"],
            alpha=config[name]["alpha"],
            momentum=config[name]["momentum"],
            eps=config[name]["eps"],
            centered=config[name]["centered"]
        )

    elif name == "Adam":
        return torch.optim.Adam(
            network.parameters(),
            lr=config[name]["lr"],
            eps=config[name]["lr_eps"],
            weight_decay=config[name]["weight_decay"],
            amsgrad=config[name]["amsgrad"],
        )
    raise ValueError("Cannot find optimizer by name {}".format(name))
