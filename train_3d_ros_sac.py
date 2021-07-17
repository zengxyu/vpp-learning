import logging
import sys
import os

import agents
import network
import field_ros_continuous
from config.config_ac import ConfigAC

from train.RosTrainer import RosTrainer

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

actor_network = network.network_ac_continuous.SAC_PolicyNet3
critic_network = network.network_ac_continuous.SAC_QNetwork3
Agent = agents.actor_critic_agents.SAC.SAC
Field = field_ros_continuous.Field

out_folder = "output_ros_random_sac"
in_folder = ""

# network
config = ConfigAC(actor_network=actor_network,
                  critic_network=critic_network,
                  out_folder=out_folder,
                  in_folder=in_folder,
                  learn_every=1,
                  console_logging_level=logging.DEBUG,
                  file_logging_level=logging.WARNING,
                  )

trainer = RosTrainer(config=config, Agent=Agent, Field=Field)

trainer.train()
