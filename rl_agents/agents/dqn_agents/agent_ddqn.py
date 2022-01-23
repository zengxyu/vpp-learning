from rl_agents.agents.dqn_agents.agent_dqn_with_fixed_Q_targets import AgentDQNWithFixedQTargets


class AgentDDQN(AgentDQNWithFixedQTargets):
    """A double DQN agent"""
    agent_name = "DDQN"

    def __init__(self, config, network_cls, action_space):
        AgentDQNWithFixedQTargets.__init__(self, config, network_cls, action_space)

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network. Double DQN
        uses the local index to pick the maximum q_value action and then the target network to calculate the q_value.
        The reasoning behind this is that it will help stop the network from overestimating q values"""
        # *next_states -> star for multiple parameters inputted
        max_action_indexes = self.q_network_local(next_states).detach().argmax(1)
        Q_targets_next = self.q_network_target(next_states).gather(1, max_action_indexes.unsqueeze(1))
        return Q_targets_next
