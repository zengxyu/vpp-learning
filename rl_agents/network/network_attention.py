#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 1/24/22 12:07 PM 
    @Description    :
        
===========================================
"""
from typing import List

import pfrl
from torch import nn
import torch
import torch.nn.functional as F


def build_cnn():
    pass


def build_mlp(
        input_dim: int,
        mlp_dims: List[int],
        activate_last_layer=False,
        activate_func=nn.ReLU(),
        last_layer_activate_func=None,
):
    """
    Build a multi-layer perceptron by given dimensions. ReLu() is added between each hidden layer.

    Parameters
    ----------
    input_dim : input dimension
    mlp_dims : list of hidden layer dimensions
    activate_last_layer : whether add ReLu activation function for the last layer
    activate_func : activation function, choose nn.ReLU() as default

    Returns
    -------
    net : built sequential model
    """
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2:
            layers.append(activate_func)
        if i == len(mlp_dims) - 2 and activate_last_layer:
            func = (
                last_layer_activate_func
                if last_layer_activate_func is not None
                else activate_func
            )
            layers.append(func)
    net = nn.Sequential(*layers)
    return net


class SpatialAttentionModel(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.n_actions = n_actions

        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)

        self.mlp_ray1 = build_mlp(input_dim=3888, mlp_dims=[1024, 512], activate_last_layer=True)

        self.mlp_ray2 = build_mlp(input_dim=512, mlp_dims=[256, 128], activate_last_layer=False)

        self.attention = build_mlp(input_dim=128, mlp_dims=[1], activate_last_layer=False)

        self.mlp_values = build_mlp(input_dim=128, mlp_dims=[], activate_last_layer=False)

    def forward(self, state):
        state = state.float()

        batch_size = state.shape[0]
        parts_size = 8
        parts = state.reshape((-1, 15, 9, 9))
        out_frame = F.relu(self.frame_con1(parts))
        out_frame = out_frame.reshape(out_frame.size()[0], -1)

        # ray_num_per_part = 15x9x9
        mlp_output1 = self.mlp_ray1(out_frame)
        # features 用来和attention的score相乘, ray_part_size 是分成多少块
        features = self.mlp_ray2(mlp_output1).view(
            batch_size, self.ray_part_size, -1
        )

        attention_scores = self.mlp_ray_attention(mlp_output1)
        attention_scores = attention_scores.view(batch_size, self.ray_part_size, 1)
        attention_scores = attention_scores.squeeze(dim=2)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2)
        weighted_feature = torch.sum(torch.mul(attention_weights, features), dim=1)
        action_values = self.mlp_values(weighted_feature)
        return pfrl.action_value.DiscreteActionValue(action_values)
