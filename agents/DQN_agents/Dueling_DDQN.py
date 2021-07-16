import torch
from torch import optim

from agents.Base_Agent_DQN import Base_Agent_DQN
from agents.DQN_agents.DDQN import DDQN


class Dueling_DDQN(DDQN):
    """A dueling double DQN agent as described in the paper http://proceedings.mlr.press/v48/wangf16.pdf
        The difference between ddqn and dueling_dqn is that the network is different,
        dueling dqn uses advantage value as the output of q_network
    """
    agent_name = "Dueling DDQN"

    def __init__(self, config):
        DDQN.__init__(self, config)
    #     self.q_network_local = self.create_NN()
    #     self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
    #                                           lr=self.hyperparameters["learning_rate"], eps=1e-4)
    #     self.q_network_target = self.create_NN()
    #     Base_Agent_DQN.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)
    #     print(self.q_network_local)
    #     self.model_dict, self.optimizer_dict = self.build_model_and_optimizer_dict()
    #
    # def build_model_and_optimizer_dict(self):
    #     model_dict = {"q_network_local": self.q_network_local,
    #                   "q_network_target": self.q_network_target}
    #     optimizer_dict = {"q_network_optimizer", self.q_network_optimizer}
    #     return model_dict, optimizer_dict
