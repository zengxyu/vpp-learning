import argparse
import logging
import sys
import os

import action_space
import agents
import network
import trainer_p3d
import environment
from config.config_dqn import ConfigDQN
from utilities.util import get_project_path, get_state_size, get_action_size

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

headless = True
if not headless:
    from direct.stdpy import threading


def train_fun():
    Network = network.network_dqn_11_temporal.DQN_Network11_Temporal_LSTM6
    Agent = agents.DQN_agents.Agent_DDQN_PER_spacial.Agent_DDQN_PER_spacial
    Field = environment.field_p3d_discrete.Field
    Action = action_space.ActionMoRoMultiplier36
    Trainer = trainer_p3d.P3DTrainer_Temporal_Pose_into_buffer.P3DTrainer
    out_folder = "out_p3d_random_env_seq_len_10_spacial"
    # in_folder = ""
    in_folder = "output_remote5/out_p3d_random_env_seq_len_10_spacial"

    # network
    config = ConfigDQN(network=Network,
                       out_folder=out_folder,
                       in_folder=in_folder,
                       learn_every=1,
                       console_logging_level=logging.DEBUG,
                       file_logging_level=logging.WARNING,
                       )

    init_file_path = os.path.join(project_path, 'VG07_6.binvox')
    max_step = 400
    seq_len = 10
    # field
    field = Field(config=config, Action=Action, shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0,
                  scale=0.05,
                  max_steps=max_step, init_file=init_file_path, headless=headless)
    config.set_parameters({"learning_rate": 5e-5})
    config.is_train = True

    config.set_parameters({"epsilon": 0.15})
    config.set_parameters({"epsilon_decay_rate": 0.985})
    config.set_parameters({"epsilon_min": 0})

    # Agent
    agent = Agent(config)
    agent.load_model(401)
    trainer = Trainer(config=config, agent=agent, field=field)
    trainer.train(is_sph_pos=False, is_randomize=True, is_global_known_map=False, is_egocetric=False,
                  is_reward_plus_unknown_cells=True,
                  randomize_control=True, is_spacial=True, seq_len=seq_len)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--info', metavar='N', type=str, nargs='+',
                    help='an integer for the accumulator')

args = parser.parse_args()
if __name__ == '__main__':
    project_path = get_project_path()

    train_fun()
