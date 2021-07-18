import sys
import os
import logging

import action_space
import agents
import network
import field_env_3d_unknown_map2_continuous

from train.P3DTrainer import P3DTrainer

from config.config_ac import ConfigAC

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

# network
actor_network = network.network_ac_continuous.SAC_PolicyNet3
critic_network = network.network_ac_continuous.SAC_QNetwork3
Agent = agents.actor_critic_agents.SAC.SAC
Field = field_env_3d_unknown_map2_continuous.Field
Action = action_space.ActionMoRoContinuous

out_folder = "output_p3d_sac"
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

trainer = P3DTrainer(config=config, Agent=Agent, Field=Field, Action=Action)

trainer.train()
