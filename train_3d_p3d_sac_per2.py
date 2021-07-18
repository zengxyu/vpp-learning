import sys
import os

import action_space
import agents
import network
import field_env_3d_unknown_map2_continuous

from train.P3DTrainer import P3DTrainer

from config.config_ac import ConfigAC

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
# network
actor_network = network.network_ac_continuous.SAC_PER_PolicyNet3
critic_network = network.network_ac_continuous.SAC_PER_QNetwork3
Agent = agents.actor_critic_agents.SAC_Prioritised_Experience_Replay.SAC_Prioritised_Experience_Replay
Field = field_env_3d_unknown_map2_continuous.Field
Action = action_space.ActionMoRoContinuous
out_folder = "output_p3d_sac_per"
in_folder = ""

# network
config = ConfigAC(actor_network=actor_network,
                  critic_network=critic_network,
                  out_folder=out_folder,
                  in_folder=in_folder,
                  learn_every=1
                  )

trainer = P3DTrainer(config=config, Agent=Agent, Field=Field, Action=Action)

trainer.train()
