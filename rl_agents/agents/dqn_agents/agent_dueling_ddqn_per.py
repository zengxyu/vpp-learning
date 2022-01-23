from rl_agents.agents.dqn_agents.agent_ddqn_per import AgentDDQNPER


class AgentDuelingDDQNPER(AgentDDQNPER):
    """A dueling double DQN agent as described in the paper http://proceedings.mlr.press/v48/wangf16.pdf"""
    agent_name = "Dueling DDQN With Prioritised Experience Replay"

    def __init__(self, config, network_cls, action_space):
        AgentDDQNPER.__init__(self, config, network_cls, action_space)
