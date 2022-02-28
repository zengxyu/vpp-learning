#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/28/22 12:50 AM 
    @Description    :
        
===========================================
"""
from agents.baseline_agents.circular_motion_agent import CircularMotionAgent
from agents.baseline_agents.random_agent import RandomAgent
from agents.policies import Policies


def build_baseline_agent(parser_args, action_space):
    if parser_args.policy == Policies.Random_Policy:
        return RandomAgent(action_space)
    elif parser_args.policy == Policies.Circular_Policy:
        return CircularMotionAgent(parser_args, action_space)
    else:
        raise NotImplementedError
