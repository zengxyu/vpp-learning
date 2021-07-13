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
        self.hyperparameters = config.hyperparameters['DQN_Agents']

        self.q_network_local = None
        self.q_network_target = None
        self.q_network_optimizer = None

    def create_NN(self):
        Network = self.config.network
        net = Network(action_size=self.config.environment['action_size']).to(self.device)
        return net

    def load_model(self, net, file_path, map_location):
        state_dict = torch.load(file_path, map_location=map_location)
        net.load_state_dict(state_dict)

    def store_model(self, model_sv_folder, i_episode):
        model_save_path = os.path.join(model_sv_folder,
                                       "Agent_dqn_state_dict_%d.mdl" % (i_episode + 1))
        torch.save(self.q_network_local.state_dict(), model_save_path)
