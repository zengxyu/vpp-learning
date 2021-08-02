import logging
import sys
import agents
import network
from config.config_ac import ConfigAC
from config.config_dqn import ConfigDQN
from train.GymTrainer import GymTrainer
from utilities.action_type import ActionType
from utilities.util import *

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


def build_ddqn_dueling():
    Network = network.network_dqn.DQN_Network_Dueling_CartPole
    Agent = agents.DQN_agents.DQN.DQN
    Field = 'CartPole-v0'
    out_folder = "output_cartpole_dueling_dqn"
    in_folder = ""
    action_type = ActionType.DISCRETE
    return Network, Agent, Field, out_folder, in_folder, action_type


Network, Agent, Field, out_folder, in_folder, action_type = build_ddqn_dueling()
# network
config = ConfigDQN(network=Network,
                   out_folder=out_folder,
                   in_folder=in_folder,
                   learn_every=1,
                   console_logging_level=logging.DEBUG,
                   file_logging_level=logging.WARNING,
                   )

trainer = GymTrainer(config=config, Agent=Agent, Field=Field, action_type=action_type)

trainer.train()
