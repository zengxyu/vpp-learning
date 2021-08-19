import torch
from torch import optim

from agents.Base_Agent_DQN import Base_Agent_DQN
from agents.DQN_agents.DDQN_PER import DDQN_PER


class Dueling_DDQN_PER(DDQN_PER):
    """A dueling double DQN agent as described in the paper http://proceedings.mlr.press/v48/wangf16.pdf"""
    agent_name = "Dueling DDQN With Prioritised Experience Replay"

    def __init__(self, config):
        DDQN_PER.__init__(self, config)
