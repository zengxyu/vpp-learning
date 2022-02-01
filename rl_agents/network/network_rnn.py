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
from torch import nn


class NetworkRNN(Recurrent, nn.Sequential):
    def __init__(self, n_actions):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
        self.frame_fc1 = torch.nn.Linear(3888, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc3 = torch.nn.Linear(128, 32)

        self.fc_val = torch.nn.Linear(32, n_actions)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, sequences, recurrent_state):
        # state = sequences.float()
        # out_frame = F.relu(self.frame_con1(state))
        # out_frame = out_frame.reshape(out_frame.size()[0], -1)
        # out_frame = F.relu(self.frame_fc1(out_frame))
        # out_frame = F.relu(self.frame_fc2(out_frame))
        #
        # out = F.relu(self.pose_fc3(out_frame))
        #
        # action_values = self.fc_val(out)
        return sequences, recurrent_state
