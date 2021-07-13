import torch
from torch import optim

from agents.Base_Agent_DQN import Base_Agent_DQN
from agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay


class Dueling_DDQN_With_Prioritised_Experience_Replay(DDQN_With_Prioritised_Experience_Replay):
    """A dueling double DQN agent as described in the paper http://proceedings.mlr.press/v48/wangf16.pdf"""
    agent_name = "Dueling DDQN With Prioritised Experience Replay"

    def __init__(self, config):
        DDQN_With_Prioritised_Experience_Replay.__init__(self, config)
        self.q_network_local = self.create_NN()
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"], eps=1e-4)
        self.q_network_target = self.create_NN()
        Base_Agent_DQN.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)
        print(self.q_network_local)
