import torch
from torch.nn import Module
from pfrl.agents import DoubleDQN
from agents.pfrl_agents.explorer_builder import get_explorer_by_name
from agents.pfrl_agents.optimizer_builder import get_optimizer_by_name
from agents.pfrl_agents.replay_buffer_builder import get_replay_buffer_by_name
from agents.pfrl_agents.scheduler_builder import get_scheduler_by_name


def build_ddqn_agent(parser_args, network: Module, action_space):
    dqn_config = parser_args.agents_config["dqn"]
    dqn_config["batch_size"] = parser_args.batch_size if parser_args.batch_size is not None else dqn_config[
        "batch_size"]
    training_config = parser_args.training_config

    optimizer = get_optimizer_by_name(parser_args, dqn_config["optimizer"], network)
    scheduler = get_scheduler_by_name(parser_args, dqn_config["scheduler"], optimizer)

    explorer = get_explorer_by_name(parser_args, dqn_config["explorer"], n_actions=action_space.n)

    replay_buffer = get_replay_buffer_by_name(parser_args, dqn_config["replay_buffer"])

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
        minibatch_size=dqn_config["batch_size"],
        recurrent=dqn_config["recurrent"]

    )
    return agent, scheduler
