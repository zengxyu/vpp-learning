import os
from typing import Dict
from pfrl.agents import DoubleDQN
from torch.nn import Module

from rl_agents.pfrl_agents.explorer_builder import get_explorer_by_name
from rl_agents.pfrl_agents.optimizer_builder import get_optimizer_by_name
from rl_agents.pfrl_agents.replay_buffer_builder import get_replay_buffer_by_name


def build_ddqn_agent(parser_args, network: Module, action_space):
    dqn_config = parser_args.agents_config["dqn"]
    training_config = parser_args.training_config

    optimizer = get_optimizer_by_name(parser_args, dqn_config["optimizer"], network)

    explorer = get_explorer_by_name(parser_args, dqn_config["explorer"], n_actions=action_space.n)

    replay_buffer = get_replay_buffer_by_name(parser_args, dqn_config["replay_buffer"])

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = DoubleDQN(
        network,
        optimizer,
        replay_buffer=replay_buffer,
        gamma=dqn_config["discount_rate"],
        explorer=explorer,
        replay_start_size=dqn_config["replay_start_size"],
        target_update_interval=dqn_config["target_update_interval"],
        update_interval=dqn_config["update_interval"],
        target_update_method=dqn_config["target_update_method"],
        gpu=training_config["gpu"],
        episodic_update_len=dqn_config["episodic_update_len"],
        recurrent=dqn_config["recurrent"]

    )
    return agent
