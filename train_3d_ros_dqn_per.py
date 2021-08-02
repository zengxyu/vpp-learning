import logging
import sys
import os

import action_space
import agents
import network
from environment import field_ros

from config.config_dqn import ConfigDQN
from train.RosTrainer import RosTrainer

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

network = network.network_dqn.DQN_Network11
Agent = agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay.DDQN_With_Prioritised_Experience_Replay
Field = field_ros.Field
Action = action_space.ActionMoRo12

out_folder = "output_ros_ddqn_per"
in_folder = ""

# network
config = ConfigDQN(network=network,
                   out_folder=out_folder,
                   in_folder=in_folder,
                   learn_every=1,
                   console_logging_level=logging.DEBUG,
                   file_logging_level=logging.WARNING,
                   )

trainer = RosTrainer(config=config, Agent=Agent, Field=Field, Action=Action)

trainer.train()
