#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/7/22 12:49 AM 
    @Description    :
        
===========================================
"""

import pfrl
import torch
from pfrl.q_functions import DiscreteActionValueHead
from torch import nn


class NetworkObsLstm(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.recurrent_sequential = pfrl.nn.RecurrentSequential(nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1),
                                                                nn.ReLU(),
                                                                nn.Flatten(),
                                                                nn.Linear(3888, 512),
                                                                nn.ReLU(),
                                                                nn.LSTM(input_size=512, hidden_size=128),
                                                                nn.Linear(128, 64),
                                                                nn.ReLU(),
                                                                nn.Linear(64, action_size),
                                                                DiscreteActionValueHead())

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state, cur):
        state = state.float()
        out = self.recurrent_sequential(state, cur)

        return out
