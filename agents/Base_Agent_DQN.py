import random
import os
import logging
import torch
import torch.optim as optim

from agents.Base_Agent import Base_Agent

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class Base_Agent_DQN(Base_Agent):
    def __init__(self, config):
        super(Base_Agent_DQN, self).__init__(config)
        self.hyper_parameters = config.hyper_parameters['DQN_Agents']

        self.q_network_local = None
        self.q_network_target = None
        self.q_network_optimizer = None

    def create_NN(self):
        Network = self.config.network
        net = Network(action_size=self.config.environment['action_size']).to(self.device)
        return net
