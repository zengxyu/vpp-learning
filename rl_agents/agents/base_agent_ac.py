import os

from rl_agents.agents.base_agent import BaseAgent

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class BaseAgentAC(BaseAgent):
    def __init__(self, config, network_cls, action_space):
        super(BaseAgentAC, self).__init__(config, network_cls, action_space)
        self.hyperparameters = config.hyper_parameters

    def create_actor_network(self, action_dim):
        ActorNetwork = self.network_cls['actor_network']
        net = ActorNetwork(action_dim=action_dim)
        return net

    def create_critic_network(self, action_dim):
        CriticNetwork = self.network_cls['critic_network']
        net = CriticNetwork(action_dim=action_dim)
        return net
