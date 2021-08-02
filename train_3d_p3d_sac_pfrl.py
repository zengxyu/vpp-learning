import logging
import sys
import os

import action_space
import agents
import field_env_3d_unknown_map2_continuous_pfrl
import field_env_3d_unknown_map2_pfrl
import network
from config.config_ac import ConfigAC

from config.config_dqn import ConfigDQN
from pfrl_src.P3DTrainer_PFRL import P3DTrainer_PFRL
from pfrl_src.P3DTrainer_PFRL_SAC import P3DTrainer_PFRL_SAC

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

if __name__ == '__main__':
    actor_network = network.network_ac_continuous.SAC_PER_PolicyNet3
    critic_network = network.network_ac_continuous.SAC_PER_QNetwork3
    Agent = agents.actor_critic_agents.SAC_Prioritised_Experience_Replay.SAC_Prioritised_Experience_Replay
    Field = field_env_3d_unknown_map2_continuous_pfrl.Field

    out_folder = "output_p3d_sac_pfrl"
    in_folder = ""
    Action = action_space.ActionMoRoContinuous

    config = ConfigAC(actor_network=actor_network,
                      critic_network=critic_network,
                      out_folder=out_folder,
                      in_folder=in_folder,
                      learn_every=1
                      )

    trainer = P3DTrainer_PFRL_SAC(config=config, Field=Field, Action=Action)

    trainer.train()
