#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/8/22 8:15 PM 
    @Description    :
        
===========================================
"""
from torch.nn import Sequential

import pfrl
import torch
import torch.nn.functional as F
from pfrl.nn import Recurrent
from pfrl.q_functions import DiscreteActionValueHead
from torch import nn


class NetworkObsVisit(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()

        self.recurrent_obs = Sequential(
            nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(3888, 512),
            nn.ReLU(),
            # nn.LSTM(input_size=512, hidden_size=128),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            DiscreteActionValueHead())

        # pfrl.nn.RecurrentSequential
        self.recurrent_visit = Sequential(
            nn.Conv3d(1, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(216, 108),
            nn.ReLU(),
            # nn.LSTM(input_size=108, hidden_size=64),
            nn.Linear(108, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
            DiscreteActionValueHead())

    def forward(self, state):
        obs = state[0].float()
        visit = state[1].float()

        out = self.recurrent_obs(obs)
        out = self.recurrent_visit(visit)
        # recurrent_obs = recurrent_state[0].float()
        # recurrent_visit = recurrent_state[1].float()
        # out_obs = self.recurrent_obs(obs, recurrent_obs)
        # out_visit = self.recurrent_visit(visit, recurrent_visit)
        return out
