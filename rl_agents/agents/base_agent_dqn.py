import os

from rl_agents.agents.base_agent import BaseAgent

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class BaseAgentDQN(BaseAgent):
    def __init__(self, config, network_cls, action_space):
        super(BaseAgentDQN, self).__init__(config, network_cls, action_space)
        self.hyper_parameters = config.hyper_parameters

        self.q_network_local = None
        self.q_network_target = None
        self.q_network_optimizer = None

    def create_network(self):
        Network = self.network_cls
        net = Network(action_size=self.action_space.n).to(self.device)
        return net
