#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/21/22 8:34 PM 
    @Description    :
        
===========================================
"""


def fix_observation_parameters(model):
    for k, v in model.named_parameters():
        if k.__contains__('recurrent_obs'):
            v.requires_grad = False  # 固定参数
