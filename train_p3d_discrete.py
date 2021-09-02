import argparse
import logging
import sys
import os

import action_space
import agents
import environment
import network
import trainer_p3d

from config.config_dqn import ConfigDQN
from utilities.util import get_project_path

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

headless = True
if not headless:
    from direct.stdpy import threading


def train_fun():
    Network = network.network_dqn_11.DQN_Network11
    Agent = agents.DQN_agents.DDQN_PER.DDQN_PER
    Field = environment.field_p3d_discrete.Field
    Action = action_space.ActionMoRo12
    Trainer = trainer_p3d.P3DTrainer.P3DTrainer
    out_folder = "out_p3d_original"
    in_folder = ""
    # network
    config = ConfigDQN(network=Network,
                       out_folder=out_folder,
                       in_folder=in_folder,
                       learn_every=1,
                       console_logging_level=logging.DEBUG,
                       file_logging_level=logging.WARNING,
                       )

    init_file_path = os.path.join(project_path, 'VG07_6.binvox')
    # field
    field = Field(config=config, Action=Action, shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0,
                  scale=0.05,
                  max_steps=300, init_file=init_file_path, headless=headless)
    config.is_train = True
    # config.set_parameters({"learning_rate": 3e-5})
    # config.set_parameters({"epsilon": 0.0})
    # config.set_parameters({"epsilon_decay_rate": 0.985})
    # config.set_parameters({"epsilon_min": 0})
    # Agent
    agent = Agent(config)
    # agent.load_model(151)
    trainer = Trainer(config=config, agent=agent, field=field)
    trainer.train(is_sph_pos=False, is_randomize=False, is_global_known_map=False, is_egocetric=False,
                  is_reward_plus_unknown_cells=False,
                  randomize_control=False, is_spacial=False, seq_len=0)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--info', metavar='N', type=str, nargs='+',
                    help='an integer for the accumulator')

args = parser.parse_args()
if __name__ == '__main__':
    project_path = get_project_path()

    train_fun()
