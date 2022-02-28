#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/6/22 5:59 PM 
    @Description    :
        
===========================================
"""
from agents.action_space.action_space import ActionMoRo10, ActionMoRo20, ActionMoRo30


def build_action_space(parser_args):
    actions = parser_args.training_config["action"]
    assert sum(actions.values()) == 1, "Only one action can be choose"
    if actions["ActionMoRo10"]:
        action_space = ActionMoRo10()
    elif actions["ActionMoRo20"]:
        action_space = ActionMoRo20()
    elif actions["ActionMoRo30"]:
        action_space = ActionMoRo30()
    else:
        raise NotImplementedError("Action : {} has not been implemented")

    return action_space
