#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 1/30/22 3:02 PM 
    @Description    :
        
===========================================
"""

import pfrl
import torch
import torch.nn.functional as F
from pfrl.nn import Recurrent
from pfrl.q_functions import DiscreteActionValueHead
from torch import nn


class NetworkVisitLstm(Recurrent, torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.recurrent_sequential = pfrl.nn.RecurrentSequential(nn.Conv3d(1, 4, kernel_size=4, stride=2, padding=1),
                                                                nn.ReLU(),
                                                                nn.Conv3d(4, 8, kernel_size=4, stride=2, padding=1),
                                                                nn.ReLU(),
                                                                nn.Flatten(),
                                                                nn.Linear(512, 256),
                                                                nn.ReLU(),
                                                                nn.LSTM(input_size=256, hidden_size=128),
                                                                nn.Linear(128, 64),
                                                                nn.ReLU(),
                                                                nn.Linear(64, action_size),
                                                                DiscreteActionValueHead())

        self.hn_neighbor_state_dim = 512
        self.lstm_neighbor1 = nn.LSTM(128, self.hn_neighbor_state_dim, batch_first=True)
        self.lstm_neighbor2 = nn.LSTM(128, self.hn_neighbor_state_dim, batch_first=True)

    def forward(self, state, cur):
        state = state.float()
        out = self.recurrent_sequential(state, cur)

        return out
