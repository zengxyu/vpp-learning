#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/28/22 12:40 AM 
    @Description    :
        
===========================================
"""
import random


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return random.randint(0, self.action_space.n - 1)
