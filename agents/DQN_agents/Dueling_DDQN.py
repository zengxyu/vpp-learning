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
