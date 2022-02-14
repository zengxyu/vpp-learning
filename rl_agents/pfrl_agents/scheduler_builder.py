#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/14/22 11:49 PM 
    @Description    :
        
===========================================
"""
import os
import torch

from config import read_yaml


def get_scheduler_by_name(parser_args, name, optimizer):
    config = read_yaml(config_dir=os.path.join(parser_args.out_folder, "configs"), config_name="scheduler.yaml")

    if name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"],
                                               last_epoch=-1)

    elif name == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["milestones"], gamma=config["gamma"],
                                                    last_epoch=-1)

    raise ValueError("Cannot find scheduler by name {}".format(name))
