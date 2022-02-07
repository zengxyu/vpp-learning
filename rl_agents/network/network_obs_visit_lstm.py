#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/7/22 1:13 AM 
    @Description    :
        
===========================================
"""

import pfrl
import torch
import torch.nn.functional as F
from pfrl.nn import Recurrent
from pfrl.q_functions import DiscreteActionValueHead
from torch import nn


class NetworkObsVisitLstm(Recurrent, torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()

        self.recurrent_obs = pfrl.nn.RecurrentSequential(
            nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3888, 512),
            nn.ReLU(),
            nn.LSTM(input_size=512, hidden_size=128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            DiscreteActionValueHead())

        self.recurrent_visit = pfrl.nn.RecurrentSequential(
            nn.Conv3d(1, 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.LSTM(input_size=512, hidden_size=128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            DiscreteActionValueHead())

    def forward(self, state, cur):
        obs = state[0].float()
        cur_obs = cur[1].float()
        visit = state[1].float()
        visit_obs = cur[1].float()
        out_obs = self.recurrent_obs(obs, cur_obs)
        out_visit = self.recurrent_visit(visit, visit_obs)
        return out_obs, out_visit
