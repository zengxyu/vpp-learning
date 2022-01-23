from rl_agents.agents.base_agent_dqn import BaseAgentDQN
# Important, 记住这里要改回来
from rl_agents.agents.dqn_agents.agent_dqn import AgentDQN
from collections import Counter


class AgentDQNWithFixedQTargets(AgentDQN):
    """A DQN agent that uses an older version of the q_network as the target network"""
    agent_name = "DQN with Fixed Q Targets"

    def __init__(self, config, network_cls, action_space):
        AgentDQN.__init__(self, config, network_cls, action_space)
        self.q_network_target = self.create_network()
        BaseAgentDQN.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)
        self.model_dict, self.optimizer_dict = self.__build_model_and_optimizer_dict()

    def learn(self):
        """Runs a learning iteration for the Q network"""
        if not self.time_for_q_network_to_learn():
            return 0

        self.q_network_local.is_train()
        states, actions, rewards, next_states, dones = self.memory.sample()  # Sample experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)

        actions_list = [action_X.item() for action_X in actions]

        self.logger.info("Action counts {}".format(Counter(actions_list)))
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss,
                                    self.hyper_parameters["gradient_clipping_norm"])
        self.skipping_step_update_of_target_network(self.q_network_local, self.q_network_target,
                                                    global_step_number=self.global_step_number,
                                                    update_every_n_steps=self.hyper_parameters[
                                                        "update_every_n_steps"])
        return loss.detach().cpu().numpy()

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def __build_model_and_optimizer_dict(self):
        model_dict = {"q_network_local": self.q_network_local,
                      "q_network_target": self.q_network_target}
        optimizer_dict = {"q_network_optimizer", self.q_network_optimizer}
        print("model_dict:", model_dict)
        return model_dict, optimizer_dict
