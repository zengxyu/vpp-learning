import logging
import sys
import agents
import network
from config.config_ac import ConfigAC
from train.GymTrainer import GymTrainer
from utilities.action_type import ActionType
from utilities.util import *

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


def build_sac():
    Agent = agents.actor_critic_agents.SAC.SAC
    Field = 'Pendulum-v0'
    Actor_network = network.network_ac_continuous.SAC_PolicyNet
    Critic_network = network.network_ac_continuous.SAC_QNetwork
    out_folder = "output_pendulum_sac_per"
    in_folder = ""
    action_type = ActionType.CONTINUOUS
    return Actor_network, Critic_network, Agent, Field, out_folder, in_folder, action_type


def build_ddpg():
    Agent = agents.actor_critic_agents.DDPG.DDPG
    Field = 'Pendulum-v0'
    Actor_network = network.network_ac_continuous.DDPG_PolicyNet
    Critic_network = network.network_ac_continuous.DDPG_QNetwork
    out_folder = "output_pendulum_ddpg"
    in_folder = ""
    action_type = ActionType.CONTINUOUS
    return Actor_network, Critic_network, Agent, Field, out_folder, in_folder, action_type


def build_td3():
    Agent = agents.actor_critic_agents.TD3.TD3
    Field = 'Pendulum-v0'
    Actor_network = network.network_ac_continuous.TD3_PolicyNet
    Critic_network = network.network_ac_continuous.TD3_QNetwork
    out_folder = "output_p3d_td3_per"
    in_folder = ""
    action_type = ActionType.CONTINUOUS
    return Actor_network, Critic_network, Agent, Field, out_folder, in_folder, action_type


Actor_network, Critic_network, Agent, Field, out_folder, in_folder, action_type = build_sac()

# network
config = ConfigAC(actor_network=Actor_network,
                  critic_network=Critic_network,
                  out_folder=out_folder,
                  in_folder=in_folder,
                  learn_every=1
                  )

trainer = GymTrainer(config=config, Agent=Agent, Field=Field, action_type=action_type)

trainer.train()
