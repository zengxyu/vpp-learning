import logging
import sys
import os

import agents
import network
import field_ros

from config.config_dqn import ConfigDQN
from train.RosRandomeTrainer import RosRandomTrainer

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

network = network.network_dqn.DQN_Network11
Agent = agents.DQN_agents.DQN.DQN
Field = field_ros.Field

out_folder = "output_ros_random_ddqn_per"
in_folder = ""

# network
config = ConfigDQN(network=network,
                   out_folder=out_folder,
                   in_folder=in_folder,
                   learn_every=1,
                   console_logging_level=logging.DEBUG,
                   file_logging_level=logging.WARNING,
                   )

trainer = RosRandomTrainer(config=config, Agent=Agent, Field=Field)

trainer.train()
