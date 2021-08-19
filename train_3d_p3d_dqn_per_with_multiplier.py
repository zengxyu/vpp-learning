import logging
import sys
import os

import action_space
import agents
import network
from environment import field_p3d_multi_discrete

from train.P3DTrainer import P3DTrainer

from config.config_dqn import ConfigDQN
from utilities.util import get_project_path

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

network = network.network_dqn.DQN_Network11
Agent = agents.DQN_agents.DDQN_PER.DDQN_PER
Field = field_p3d_multi_discrete.Field
Action = action_space.ActionMoRo15

out_folder = "output_p3d_ddqn_per_with_multiplier"
in_folder = ""

# network
config = ConfigDQN(network=network,
                   out_folder=out_folder,
                   in_folder=in_folder,
                   learn_every=1,
                   console_logging_level=logging.DEBUG,
                   file_logging_level=logging.WARNING,
                   )

trainer = P3DTrainer(config=config, Agent=Agent, Field=Field, Action=Action, project_path=get_project_path())

trainer.train()
