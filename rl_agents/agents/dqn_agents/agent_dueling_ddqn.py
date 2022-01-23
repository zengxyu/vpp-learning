from rl_agents.agents.dqn_agents.agent_ddqn import AgentDDQN


class AgentDuelingDDQN(AgentDDQN):
    """A dueling double DQN agent as described in the paper http://proceedings.mlr.press/v48/wangf16.pdf
        The difference between ddqn and dueling_dqn is that the network is different,
        dueling dqn uses advantage value as the output of q_network
    """
    agent_name = "Dueling DDQN"

    def __init__(self, config, network_cls, action_space):
        AgentDDQN.__init__(self, config, network_cls, action_space)
