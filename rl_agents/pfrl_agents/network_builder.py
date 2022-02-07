#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/6/22 6:06 PM 
    @Description    :
        
===========================================
"""
from rl_agents.network.network_attention import SpatialAttentionModel

from rl_agents.network.network_obs import NetworkObs
from rl_agents.network.network_obs_lstm import NetworkObsTemporal
from rl_agents.network.network_visit import NetworkVisit

from rl_agents.network.network_visit_temporal import NetworkVisitTemporal


def build_network(parser_args, action_size):
    networks = parser_args.training_config["network"]
    assert sum(networks.values()) == 1, "Only one network can be choose"

    if networks["NetworkObs"]:
        network = NetworkObs(action_size)
    elif networks["NetworkObsTemporal"]:
        network = NetworkObsTemporal(action_size)
    elif networks["NetworkVisit"]:
        network = NetworkVisit(action_size)
    elif networks["NetworkVisitTemporal"]:
        network = NetworkVisitTemporal(action_size)
    elif networks["SpatialAttentionModel"]:
        network = SpatialAttentionModel(action_size)

    # elif networks["NetworkRNN"]:
    #     network = NetworkRNN(n_actions=action_size)
    else:
        raise NotImplementedError("Action : {} has not been implemented")

    return network
