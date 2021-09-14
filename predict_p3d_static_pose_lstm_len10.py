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
    Action = action_space.ActionMoRoMultiplier36
    Trainer = trainer_p3d.P3DTrainer_Temporal.P3DTrainer
    out_folder = "out_p3d_static_step_len_10_36_action_predict_model_550"
    in_folder = "output/out_p3d_random_step_len_10_36_action_with_scheduler"
    # network
    config = ConfigDQN(network=Network,
                       out_folder=out_folder,
                       in_folder=in_folder,
                       learn_every=1,
                       console_logging_level=logging.DEBUG,
                       file_logging_level=logging.WARNING,
                       )
    config.is_train = False
    config.set_parameters({"learning_rate": 8e-6})
    config.set_parameters({"epsilon": 0.1})
    config.set_parameters({"epsilon_min": 0.1})
    config.num_episodes_to_run = 50

    init_file_path = os.path.join(project_path, 'VG07_6.binvox')
    max_step = 400
    seq_len = 10
    # field
    field = Field(config=config, Action=Action, shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0,
                  scale=0.05,
                  max_steps=max_step, init_file=init_file_path, headless=headless)
    # map_path = "/Users/weixianshi/PycharmProjects/vpp-learning/output/out_36_envs/global_map_42.obj"
    # field.global_map = pickle.load(open(map_path, "rb"))
    # Agent
    agent = Agent(config, is_add_revisit_map=False)
    agent.load_model(550)
    trainer = Trainer(config=config, agent=agent, field=field)
    trainer.train(is_randomize=False, randomize_control=False, randomize_from_48_envs=False,
                  is_reward_plus_unknown_cells=True, seq_len=seq_len,
                  is_add_negative_reward=False, is_map_diff_reward=False, is_stop_n_zero_rewards=False,
                  is_save_path=False, is_save_env=False)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--info', metavar='N', type=str, nargs='+',
                    help='an integer for the accumulator')

args = parser.parse_args()
if __name__ == '__main__':
    project_path = get_project_path()

    train_fun()
