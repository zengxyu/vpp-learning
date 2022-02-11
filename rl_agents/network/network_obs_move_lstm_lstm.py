#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/11/22 6:39 PM 
    @Description    :
        
===========================================
"""
# !/usr/bin/env python
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

from pfrl.utils.recurrent import unwrap_packed_sequences_recursive, pack_sequences_recursive


class NetworkObsMoveLstmLstm(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()

        self.recurrent_obs = pfrl.nn.RecurrentSequential(
            nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(1728, 512),
            nn.ReLU(),
            nn.LSTM(input_size=512, hidden_size=256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
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
        obs = state[0].float()
        relative_move = state[1].float()
        recurrent_state0 = recurrent_state[0] if recurrent_state is not None else None
        recurrent_state1 = recurrent_state[1] if recurrent_state is not None else None

        out_obs, recurrent_obs = self.recurrent_obs(obs, recurrent_state0)
        out_move, recurrent_visit = self.recurrent_move(relative_move, recurrent_state1)

        out_obs = unwrap_packed_sequences_recursive(out_obs)
        out_move = unwrap_packed_sequences_recursive(out_move)
        out = torch.cat((out_obs, out_move), dim=1)
        out = F.relu(self.fc1(out))
        action_values = self.fc2(out)

        return pfrl.action_value.DiscreteActionValue(action_values), (recurrent_obs, recurrent_visit)
