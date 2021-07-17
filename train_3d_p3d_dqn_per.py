import logging
import sys
import os

import agents
import network
import field_env_3d_unknown_map2

from train.P3DTrainer import P3DTrainer

from config.config_dqn import ConfigDQN

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

network = network.network_dqn.DQN_Network11
Agent = agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay.DDQN_With_Prioritised_Experience_Replay
Field = field_env_3d_unknown_map2.Field

out_folder = "output_p3d_ddqn_per"
in_folder = ""

# network
config = ConfigDQN(network=network,
                   out_folder=out_folder,
                   in_folder=in_folder,
                   learn_every=1,
                   console_logging_level=logging.DEBUG,
                   file_logging_level=logging.WARNING,
                   )

trainer = P3DTrainer(config=config, Agent=Agent, Field=Field)

trainer.train()
