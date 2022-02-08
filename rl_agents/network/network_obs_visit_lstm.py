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
from torch.nn import Sequential, ReLU

import pfrl
import torch
import torch.nn.functional as F
from pfrl.nn import Recurrent
from pfrl.q_functions import DiscreteActionValueHead
from torch import nn

from pfrl.utils.recurrent import unwrap_packed_sequences_recursive


class Expand(nn.Module):
    def forward(self, x):
        if x.shape[0] != 1:
            x = x.reshape(5, 2, -1)
        return x


class NetworkObsVisitLstm(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()

        self.recurrent_obs = Sequential(
            nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(3888, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU())

        # pfrl.nn.RecurrentSequential
        self.recurrent_visit = pfrl.nn.RecurrentSequential(
            nn.Conv3d(1, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(216, 108),
            nn.ReLU(),
            nn.LSTM(input_size=108, hidden_size=96),
            nn.Linear(96, 64),
            nn.ReLU())

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, action_size)

    def forward(self, state, recurrent_state):
        obs = state[0].data.float()
        # print("obs shape:{}".format(obs.shape))
        out_obs = self.recurrent_obs(obs)

        visit = state[1].float()
        # print("visit shape:{}".format(visit.data.shape))
        out_visit, recurrent_visit = self.recurrent_visit(visit, recurrent_state)

        out_visit = unwrap_packed_sequences_recursive(out_visit)
        out = torch.cat((out_obs, out_visit), dim=1)
        out = F.relu(self.fc1(out))
        action_values = self.fc2(out)

        return pfrl.action_value.DiscreteActionValue(action_values), recurrent_visit
