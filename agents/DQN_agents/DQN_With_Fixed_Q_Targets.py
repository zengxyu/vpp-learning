import copy

from agents.Base_Agent_DQN import Base_Agent_DQN
from agents.DQN_agents.DQN import DQN


class DQN_With_Fixed_Q_Targets(DQN):
    """A DQN agent that uses an older version of the q_network as the target network"""
    agent_name = "DQN with Fixed Q Targets"

    def __init__(self, config):
        DQN.__init__(self, config)
        self.q_network_target = self.create_NN()
        Base_Agent_DQN.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)
        self.model_dict, self.optimizer_dict = self.__build_model_and_optimizer_dict()

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        loss = super(DQN_With_Fixed_Q_Targets, self).learn(experiences=experiences)
        self.soft_update_of_target_network(self.q_network_local, self.q_network_target,
                                           self.hyper_parameters["tau"])  # Update the target network
        return loss

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def __build_model_and_optimizer_dict(self):
        model_dict = {"q_network_local": self.q_network_local,
                      "q_network_target": self.q_network_target}
        optimizer_dict = {"q_network_optimizer", self.q_network_optimizer}
        return model_dict, optimizer_dict
