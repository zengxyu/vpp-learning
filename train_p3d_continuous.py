import sys
import os
import logging

import action_space
import agents
import network

from train.P3DTrainer import P3DTrainer

from config.config_ac import ConfigAC
from environment import field_p3d_continuous

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


def build_ddpg():
    # network
    Actor_network = network.network_ac_continuous.DDPG_PolicyNet3
    Critic_network = network.network_ac_continuous.DDPG_QNetwork3
    Agent = agents.actor_critic_agents.DDPG.DDPG
    Field = field_p3d_continuous.Field
    Action = action_space.ActionMoRoContinuous

    out_folder = "output_p3d_ddpg"
    in_folder = ""
    return Actor_network, Critic_network, Agent, Field, Action, out_folder, in_folder


def build_sac():
    # network
    Actor_network = network.network_ac_continuous.SAC_PolicyNet3
    Critic_network = network.network_ac_continuous.SAC_QNetwork3
    Agent = agents.actor_critic_agents.SAC.SAC
    Field = field_p3d_continuous.Field
    Action = action_space.ActionMoRoContinuous

    out_folder = "output_p3d_sac"
    in_folder = ""
    return Actor_network, Critic_network, Agent, Field, Action, out_folder, in_folder


def build_sac_per():
    # network
    Actor_network = network.network_ac_continuous.SAC_PER_PolicyNet3
    Critic_network = network.network_ac_continuous.SAC_PER_QNetwork3
    Agent = agents.actor_critic_agents.SAC_Prioritised_Experience_Replay.SAC_Prioritised_Experience_Replay
    Field = field_p3d_continuous.Field
    Action = action_space.ActionMoRoContinuous
    out_folder = "output_p3d_sac_per"
    in_folder = ""
    return Actor_network, Critic_network, Agent, Field, Action, out_folder, in_folder


def build_td3():
    # network
    Actor_network = network.network_ac_continuous.TD3_PolicyNet3
    Critic_network = network.network_ac_continuous.TD3_QNetwork3
    Agent = agents.actor_critic_agents.TD3.TD3
    Field = field_p3d_continuous.Field
    Action = action_space.ActionMoRoContinuous

    out_folder = "output_p3d_td3"
    in_folder = ""
    return Actor_network, Critic_network, Agent, Field, Action, out_folder, in_folder


Actor_network, Critic_network, Agent, Field, Action, out_folder, in_folder = build_sac()

# network
config = ConfigAC(actor_network=Actor_network,
                  critic_network=Critic_network,
                  out_folder=out_folder,
                  in_folder=in_folder,
                  learn_every=1,
                  console_logging_level=logging.DEBUG,
                  file_logging_level=logging.WARNING,
                  )

trainer = P3DTrainer(config=config, Agent=Agent, Field=Field, Action=Action)

trainer.train()
