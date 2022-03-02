#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/9/22 2:27 AM 
    @Description    :
        
===========================================
"""

import pfrl
import torch
import torch.nn.functional as F
from torch.nn import Sequential
from torch import nn

from pfrl.utils.recurrent import unwrap_packed_sequences_recursive


class NetworkObsMoveLstmDeeperObs_48_24_8(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()

        self.recurrent_obs = Sequential(
            nn.Conv2d(32, 40, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(40, 56, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(56, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.recurrent_move = pfrl.nn.RecurrentSequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LSTM(input_size=128, hidden_size=96),
            nn.Linear(96, 64),
            nn.ReLU())

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, state, recurrent_state):
        obs = state[0].data.float()
        # print("obs shape:{}".format(obs.shape))
        out_obs = self.recurrent_obs(obs)

        relative_move = state[1].float()
        # print("visit shape:{}".format(visit.data.shape))
        out_move, recurrent_visit = self.recurrent_move(relative_move, recurrent_state)

        out_move = unwrap_packed_sequences_recursive(out_move)
        out = torch.cat((out_obs, out_move), dim=1)
        out = F.relu(self.fc1(out))
        action_values = self.fc2(out)

        return pfrl.action_value.DiscreteActionValue(action_values), recurrent_visit
