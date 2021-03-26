import sys
import os
import argparse

from scipy.spatial.transform.rotation import Rotation

from agent.agent_dqn3 import Agent
from field_env_3d_known_map import Action
from field_env_3d_unknown_map import Field
from network.network_dqn import DQN_Network6, DQN_Network8, DQN_Network9
from util.summary_writer import MySummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=True, action="store_true", help="Run in headless mode")
args = parser.parse_args()
if not args.headless:
    from direct.stdpy import threading

"""
出发点 出发方向固定
"""

params = {
    'name': 'dqn',

    # model params
    'update_every': 10,
    'eps_start': 0.15,  # Default/starting value of eps
    'eps_decay': 0.99999,  # Epsilon decay rate
    'eps_min': 0.15,  # Minimum epsilon
    'gamma': 0.9,
    'buffer_size': 200000,
    'batch_size': 128,
    'action_size': len(Action),

    'is_double': False,
    'is_priority_buffer': True,

    # grid params
    'max_step': 200,

    # train params
    'is_train': True,
    'visualise': True,
    'is_normalize': False,
    'num_episodes': 5000000,
    'scale': 15,
    'use_gpu': True,
    'model': DQN_Network9,

    # folder params

    # output
    'output_folder': "output_dqn6",
    'log_folder': 'log',
    'model_folder': 'model',
    'memory_config_dir': "memory_config"

}

params['log_folder'] = os.path.join(params['output_folder'], params['log_folder'])
params['model_folder'] = os.path.join(params['output_folder'], params['model_folder'])
if not os.path.exists(params['log_folder']):
    os.makedirs(params['log_folder'])
if not os.path.exists(params['model_folder']):
    os.makedirs(params['model_folder'])

# model_path = os.path.join(params['output_folder'], "model", "Agent_dqn_state_dict_1600.mdl")
# model_path = os.path.join("output_dqn", "model", "Agent_dqn_state_dict_123600.mdl")

log_dir = os.path.join(params['output_folder'], 'log')
summary_writer = MySummaryWriter(log_dir)

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=300,
              init_file='VG07_6.binvox', headless=args.headless)
player = Agent(params, summary_writer)

all_mean_rewards = []
all_mean_losses = []
time_step = 0

observed_map, robot_pose = field.reset()
initial_direction = np.array([[-1], [0], [0]])

for i_episode in range(params['num_episodes']):
    done = False
    # rewards = []
    rewards1 = []
    actions = []

    # rewards2 = []

    while not done:
        action = player.act(observed_map, robot_pose)
        observed_map_next, robot_pose_next, reward1, reward3, done = field.step(action)

        # robot direction
        robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ initial_direction
        robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ initial_direction

        # diff direction
        robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)
        robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)

        player.step(state=[observed_map, robot_pose_input], action=action, reward=reward1,
                    next_state=[observed_map_next, robot_pose_input_next], done=done)

        # 转到下一个状态
        observed_map = observed_map_next.copy()
        robot_pose = robot_pose_next.copy()
        # train
        loss = player.learn(memory_config_dir=params['memory_config_dir'])

        time_step += 1
        # record
        summary_writer.add_loss(loss)
        summary_writer.add_reward(reward1, i_episode)

        actions.append(action)
        rewards1.append(reward1)
        # rewards2.append(reward3)

        if not args.headless:
            threading.Thread.considerYield()

        # rewards.append(reward)
        if done:

            print("\nepisode {} over".format(i_episode))
            print("mean rewards1:{}".format(np.sum(rewards1)))
            print("robot pose: {}".format(robot_pose[:3]))
            print("actions:{}".format(np.array(actions)))
            print("rewards:{}".format(np.array(rewards1)))
            # print("mean rewards2:{}; new visit cell num: {}".format(np.sum(rewards2), np.sum(rewards2) / r_ratio))
            player.reset()
            observed_map, robot_pose = field.reset()
            rewards1 = []
            rewards2 = []

            if (i_episode + 1) % 200 == 0:
                # plt.cla()
                model_save_path = os.path.join(params['model_folder'], "Agent_dqn_state_dict_%d.mdl" % (i_episode + 1))
                player.store_model(model_save_path)

print('Complete')
