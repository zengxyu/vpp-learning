import logging
import sys
import os

import agents
import network
from environment import field_ros
import action_space
from config.config_dqn import ConfigDQN
from train.RosRandomeTrainer import RosRandomTrainer
from train.RosTrainer import RosTrainer
from utilities.data_structures.Constant import EpsExplorationStrategy

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

network = network.network_dqn.DQN_Network11
Agent = agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay.DDQN_With_Prioritised_Experience_Replay
Field = field_ros.Field
Action = action_space.ActionMoRo12

out_folder = "output_reset_and_random9"
in_folder = "output_reset_and_random6"

eps_exploration_strategy_config = {EpsExplorationStrategy.INVERSE_STRATEGY: {"epsilon": 1.0,
                                                                             'epsilon_decay_denominator': 1.0},
                                   EpsExplorationStrategy.EXPONENT_STRATEGY: {"epsilon": 0.5,
                                                                              "epsilon_decay_rate": 0.997,
                                                                              "epsilon_min": 0.1},
                                   EpsExplorationStrategy.CYCLICAL_STRATEGY: {"exploration_cycle_episodes_length": 100}
                                   }
# network
config = ConfigDQN(network=network,
                   out_folder=out_folder,
                   in_folder=in_folder,
                   learn_every=1,
                   console_logging_level=logging.DEBUG,
                   file_logging_level=logging.WARNING,
                   eps_exploration_strategy_config=eps_exploration_strategy_config
                   )
config.is_train = False
if config.is_train:
    trainer = RosRandomTrainer(config=config, Agent=Agent, Field=Field, Action=Action)
else:
    trainer = RosTrainer(config=config, Agent=Agent, Field=Field, Action=Action)

trainer.agent.load_model(index=49)
trainer.train()
