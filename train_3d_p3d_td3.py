import sys
import os

import agents
import network
import field_env_3d_unknown_map2_continuous

from train.P3DTrainer import P3DTrainer

from config.config_ac import ConfigAC

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

# network
actor_network = network.network_ac_continuous.TD3_PolicyNet3
critic_network = network.network_ac_continuous.TD3_QNetwork3
Agent = agents.actor_critic_agents.TD3.TD3
Field = field_env_3d_unknown_map2_continuous.Field

out_folder = "output_p3d_td3"
in_folder = ""

# network
config = ConfigAC(actor_network=actor_network,
                  critic_network=critic_network,
                  out_folder=out_folder,
                  in_folder=in_folder,
                  learn_every=1
                  )

trainer = P3DTrainer(config=config, Agent=Agent, Field=Field)

trainer.train()
