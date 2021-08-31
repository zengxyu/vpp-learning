import argparse
import logging
import sys
import os

import action_space
import agents
import network
from environment import field_p3d_discrete

from trainer_p3d.P3DTrainer_Knownmap import P3DTrainer

from config.config_dqn import ConfigDQN
from utilities.util import get_project_path

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


def build_ddqn_per():
    Network = network.network_known_map.DQN_Network11_KnownMap
    Agent = agents.DQN_agents.Agent_DDQN_PER_KnownMap.Agent_DDQN_PER_KnownMap
    Field = field_p3d_discrete.Field
    Action = action_space.ActionMoRo12

    out_folder = "output_p3d_known_map"
    in_folder = ""

    return Network, Agent, Field, Action, out_folder, in_folder


def build_ddqn_dueling_per():
    Network = network.network_dqn.DQN_Network11_Dueling
    Agent = agents.DQN_agents.Dueling_DDQN_PER.Dueling_DDQN_PER
    Field = field_p3d_discrete.Field
    Action = action_space.ActionMoRo15

    out_folder = "output_p3d_ddqn_dueling"
    in_folder = ""
    return Network, Agent, Field, Action, out_folder, in_folder


def build_ddqn_per_without_robotpose():
    Network = network.network_dqn.DQN_Network11_Without_RobotPose
    Agent = agents.DQN_agents.DDQN_PER.DDQN_PER
    Field = field_p3d_discrete.Field
    Action = action_space.ActionMoRo12

    out_folder = "output_p3d_ddqn_per_without_robotpose"
    in_folder = ""

    return Network, Agent, Field, Action, out_folder, in_folder


def train_fun(tuning_param):
    Network, Agent, Field, Action, out_folder, in_folder = build_ddqn_per()

    # network
    config = ConfigDQN(network=Network,
                       out_folder=out_folder,
                       in_folder=in_folder,
                       learn_every=1,
                       console_logging_level=logging.DEBUG,
                       file_logging_level=logging.WARNING,
                       )
    trainer = P3DTrainer(config=config, Agent=Agent, Field=Field, Action=Action,
                         project_path=tuning_param["project_path"])
    trainer.train()


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--info', metavar='N', type=str, nargs='+',
                    help='an integer for the accumulator')

args = parser.parse_args()
if __name__ == '__main__':
    project_path = get_project_path()

    train_fun({"project_path": project_path})
