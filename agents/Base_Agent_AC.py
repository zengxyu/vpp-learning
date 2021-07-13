import os

from agents.Base_Agent import Base_Agent

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class Base_Agent_AC(Base_Agent):
    def __init__(self, config):
        super(Base_Agent_AC, self).__init__(config)
        self.hyperparameters = config.hyperparameters['Actor_Critic_Agents']

    def create_actor_network(self, state_dim, action_dim, output_dim):
        ActorNetwork = self.config.actor_network
        net = ActorNetwork(state_dim=state_dim, action_dim=action_dim)
        return net

    def create_critic_network(self, state_dim, action_dim, output_dim):
        CriticNetwork = self.config.critic_network
        net = CriticNetwork(state_dim=state_dim, action_dim=action_dim)
        return net
