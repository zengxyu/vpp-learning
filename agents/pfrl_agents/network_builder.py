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

from agents.network.network_obs import NetworkObs
from agents.network.network_obs_lstm import NetworkObsLstm
from agents.network.network_obs_move_lstm import NetworkObsMoveLstm
from agents.network.network_obs_move_lstm_deeper_obs import NetworkObsMoveLstmDeeperObs
from agents.network.network_obs_move_lstm_deeper_obs_48_24_8 import NetworkObsMoveLstmDeeperObs_48_24_8
from agents.network.network_obs_move_lstm_lstm import NetworkObsMoveLstmLstm
from agents.network.network_obs_visit import NetworkObsVisit
from agents.network.network_obs_visit_lstm import NetworkObsVisitLstm
from agents.network.network_visit import NetworkVisit
from agents.network.network_visit_lstm import NetworkVisitLstm
from agents.network.network_attention import SpatialAttentionModel


def build_network(parser_args, action_size):
    networks = parser_args.training_config["network"]
    assert sum(networks.values()) == 1, "Only one network can be choose!"

    if networks["NetworkObs"]:
        network = NetworkObs(action_size)
    elif networks["NetworkObsLstm"]:
        network = NetworkObsLstm(action_size)
    elif networks["NetworkVisit"]:
        network = NetworkVisit(action_size)
    elif networks["NetworkVisitLstm"]:
        network = NetworkVisitLstm(action_size)
    elif networks["NetworkObsVisitLstm"]:
        network = NetworkObsVisitLstm(action_size)
    elif networks["NetworkObsVisit"]:
        network = NetworkObsVisit(action_size)
    elif networks["NetworkObsMoveLstm"]:
        network = NetworkObsMoveLstm(action_size)
    elif networks["NetworkObsMoveLstmLstm"]:
        network = NetworkObsMoveLstmLstm(action_size)
    elif networks["NetworkObsMoveLstmDeeperObs"]:
        network = NetworkObsMoveLstmDeeperObs(action_size)
    elif networks["NetworkObsMoveLstmDeeperObs_48_24_8"]:
        network = NetworkObsMoveLstmDeeperObs_48_24_8(action_size)
    elif networks["SpatialAttentionModel"]:
        network = SpatialAttentionModel(action_size)
    else:
        raise NotImplementedError("Action : {} has not been implemented!")
    print("Network structure:{}".format(network))
    return network
