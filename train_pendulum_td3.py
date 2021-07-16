import logging
import sys
import agents
import network
from config.config_ac import ConfigAC
from train.GymTrainer import GymTrainer
from utilities.action_type import ActionType
from utilities.util import *

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

actor_network = network.network_ac_continuous.TD3_PolicyNet
critic_network = network.network_ac_continuous.TD3_QNetwork
out_folder = "output_p3d_td3_per"
in_folder = ""
learn_every = 1
action_type = ActionType.CONTINUOUS

# network
config = ConfigAC(actor_network=actor_network,
                  critic_network=critic_network,
                  out_folder=out_folder,
                  in_folder=in_folder,
                  learn_every=1,
                  )

Agent = agents.actor_critic_agents.TD3.TD3
Field = 'Pendulum-v0'

trainer = GymTrainer(config=config, Agent=Agent, Field=Field, action_type=action_type)

trainer.train()
