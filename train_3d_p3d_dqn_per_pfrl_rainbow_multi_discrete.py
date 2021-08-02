import logging
import sys
import os

import action_space
from environment import field_p3d_multi_discrete
import network

from config.config_dqn import ConfigDQN
from train_pfrl.P3DTrainer_PFRL_Rainbow import P3DTrainer_PFRL

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

if __name__ == '__main__':
    # Agent = agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay.DDQN_With_Prioritised_Experience_Replay
    Field = field_p3d_multi_discrete.Field
    # Action = action_space.ActionMoRo12

    out_folder = "output_p3d_ddqn_per_rainbow4"
    in_folder = ""
    Action = action_space.ActionMoRo15

    # network
    config = ConfigDQN(network=network,
                       out_folder=out_folder,
                       in_folder=in_folder,
                       learn_every=1,
                       console_logging_level=logging.DEBUG,
                       file_logging_level=logging.WARNING,
                       )

    trainer = P3DTrainer_PFRL(config=config, Field=Field, Action=Action)

    trainer.train()
