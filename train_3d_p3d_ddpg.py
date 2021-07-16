import sys
import os
import logging

import agents
import network
import field_env_3d_unknown_map2_continuous

from train.P3DTrainer import P3DTrainer

from config.config_ac import ConfigAC

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

"""不收敛 : DDPG奖励全是0"""

# network
actor_network = network.network_ac_continuous.DDPG_PolicyNet3
critic_network = network.network_ac_continuous.DDPG_QNetwork3
Agent = agents.actor_critic_agents.DDPG.DDPG
Field = field_env_3d_unknown_map2_continuous.Field

out_folder = "output_p3d_ddpg"
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

trainer = P3DTrainer(config=config, Agent=Agent, Field=Field)

trainer.train()
