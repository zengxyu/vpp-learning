import argparse
import logging
import pickle
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
    Network = network.network_dqn_11_temporal.DQN_Network11_Temporal_LSTM3
    Agent = agents.DQN_agents.DDQN_PER.DDQN_PER
    Field = environment.field_p3d_discrete.Field
    Action = action_space.ActionMoRo12
    Trainer = trainer_p3d.P3DTrainer_Temporal.P3DTrainer
    out_folder = "out_36_envs"
    in_folder = ""
    # network
    config = ConfigDQN(network=Network,
                       out_folder=out_folder,
                       in_folder=in_folder,
                       learn_every=1,
                       console_logging_level=logging.DEBUG,
                       file_logging_level=logging.WARNING,
                       )
    seq_len = 10
    max_steps = 400
    init_file_path = os.path.join(project_path, 'VG07_6.binvox')

    # field
    field = Field(config=config, Action=Action, shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0,
                  scale=0.05, max_steps=max_steps, init_file=init_file_path, headless=headless)
    num_envs = 48
    for i in range(num_envs):
        global_map = field.augment_env()
        pickle.dump(global_map, open(
            os.path.join(config.folder['out_folder'], "global_map_{}.obj".format(i)), "wb"))
        print("save global map {} to local".format(i))
    # Agent
    # agent = Agent(config, is_add_revisit_map=False)

    # trainer = Trainer(config=config, agent=agent, field=field)
    # trainer.train(is_randomize=True, randomize_control=True, is_reward_plus_unknown_cells=True, seq_len=seq_len,
    #               is_add_negative_reward=False, is_map_diff_reward=False, is_stop_n_zero_rewards=False,
    #               is_save_path=False, is_save_env=False)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--info', metavar='N', type=str, nargs='+',
                    help='an integer for the accumulator')

args = parser.parse_args()
if __name__ == '__main__':
    project_path = get_project_path()

    train_fun()
