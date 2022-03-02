#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/6/22 8:25 PM 
    @Description    :
        
===========================================
"""

import torch
from pfrl.q_functions import DiscreteActionValueHead
from torch import nn


class NetworkVisit(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.sequential = nn.Sequential(nn.Conv3d(1, 4, kernel_size=4, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.Conv3d(4, 8, kernel_size=4, stride=2,  padding=1),
                                        nn.ReLU(),
                                        nn.Flatten(),
                                        nn.Linear(4096, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, action_size),
                                        DiscreteActionValueHead())

    def forward(self, state):
        state = state.float()
        out = self.sequential(state)
        return out
