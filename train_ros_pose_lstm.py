import sys
import os

import trainer_ros

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from environment import field_ros

import argparse
import logging
import sys
import os

import action_space
import agents
import network
from config.config_dqn import ConfigDQN
from utilities.util import get_project_path

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


def train_fun():
    Network = network.network_dqn_11_temporal.DQN_Network11_Temporal_LSTM3
    Agent = agents.DQN_agents.DDQN_PER.DDQN_PER
    Field = field_ros.Field
    Action = action_space.ActionMoRoMultiplier36
    Trainer = trainer_ros.RosTrainer_Temporal.RosTrainer
    out_folder = "out_ros_static_env_pose_lstm"
    in_folder = ""
    # network
    config = ConfigDQN(network=Network,
                       out_folder=out_folder,
                       in_folder=in_folder,
                       learn_every=1,
                       console_logging_level=logging.DEBUG,
                       file_logging_level=logging.WARNING,
                       )

    max_step = 300
    seq_len = 10
    # field
    # 原本move_step = 1
    field = Field(config=config, Action=Action, shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0,
                  max_steps=max_step, move_step=0.1, handle_simulation=False)
    config.set_parameters({"learning_rate": 1e-4})
    # config.set_parameters({"epsilon": 0.4})
    # config.set_parameters({"epsilon_decay_rate": 0.985})
    # config.set_parameters({"epsilon_min": 0.01})
    # Agent
    agent = Agent(config, is_add_revisit_map=False)

    trainer = Trainer(config=config, agent=agent, field=field)
    trainer.train(is_sph_pos=False, is_global_known_map=False, is_egocetric=False,
                  is_randomize=False, randomize_control=False, seq_len=seq_len)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--info', metavar='N', type=str, nargs='+',
                    help='an integer for the accumulator')

args = parser.parse_args()
if __name__ == '__main__':
    project_path = get_project_path()

    train_fun()
