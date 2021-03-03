import sys
import os
import argparse
from agent.agent_dqn import Agent
from field_env_3d_unknown_map import Field,Action
from network.network_dqn import DQN_Network5
from util.summary_writer import MySummaryWriter
from util.util import get_euclidean_distance

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import numpy as np
"""
起始位置为随机
改了network,DQN_Network5
继续改reward
"""
parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=True, action="store_true", help="Run in headless mode")
args = parser.parse_args()
if not args.headless:
    from direct.stdpy import threading

params = {
    'name': 'dqn',

    # model params
    'update_every': 20,
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
    'max_step': 1000,
    'model': DQN_Network5,

    # train params
    'is_train': True,
    'visualise': True,
    'is_normalize': False,
    'num_episodes': 5000000,
    'scale': 15,
    'use_gpu': True,

    # folder params

    # output
    'output_folder': "output_dqn9",
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

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=200,
              init_file='VG07_6.binvox', headless=args.headless)
player = Agent(params, summary_writer)

all_mean_rewards = []
all_mean_losses = []
time_step = 0

observed_map, robot_pose = field.reset()
distances_travelled = []
is_closer_list = []

for i_episode in range(params['num_episodes']):
    done = False
    # rewards = []
    rewards1 = []
    time_step = 0
    rewards2 = []

    destination = np.array([128,128,128])
    init_observed_map, init_robot_pose = observed_map, robot_pose
    actions = []

    while not done:
        action = player.act(observed_map, robot_pose)
        observed_map_next, robot_pose_next, reward1, reward3, done = field.step(action)
        # 这里构造奖励
        reward2 = 10 - get_euclidean_distance(robot_pose_next[:3], destination)
        reward2 = int(reward2)

        reward = reward2
        robot_pose_input_next = np.append(destination-robot_pose_next[:3], robot_pose_next)

        player.step(state=[observed_map, robot_pose], action=action, reward=reward,
                    next_state=[observed_map_next, robot_pose_next], done=done)

        # train
        if time_step % 2 == 0:
            loss = player.learn(memory_config_dir=params['memory_config_dir'])
            summary_writer.add_loss(loss)

        time_step += 1
        # record
        summary_writer.add_reward(reward1, i_episode)
        actions.append(action)
        rewards1.append(reward1)
        rewards2.append(reward2)
        # 转到下一个状态
        observed_map = observed_map_next.copy()
        robot_pose = robot_pose_next.copy()

        if not args.headless:
            threading.Thread.considerYield()
        done = done or get_euclidean_distance(robot_pose_next[:3], destination) == 0
        # rewards.append(reward)
        if done:
            end_observed_map, end_robot_pose = observed_map, robot_pose
            distance_travelled = get_euclidean_distance(end_robot_pose[:3], init_robot_pose[:3])
            distances_travelled.append(distance_travelled)
            print("\nepisode {} over".format(i_episode))
            print("mean rewards1:{}".format(np.sum(rewards1)))
            print("mean rewards2:{}".format(np.sum(rewards2)))
            print("distance travelled:{}".format(distance_travelled))
            print("max distance travelled:{}".format(np.max(distances_travelled)))
            print("rewards2:{}".format(rewards2))
            print("actions:{}".format(actions))

            # print("mean rewards2:{}; new visit cell num: {}".format(np.sum(rewards2), np.sum(rewards2) / r_ratio))
            is_closer = get_euclidean_distance(end_robot_pose[:3], destination) < get_euclidean_distance(
                init_robot_pose[:3], destination)
            is_closer_list.append(is_closer)
            print("in this episode, robot travels from {} to {}".format(init_robot_pose[:3], end_robot_pose[:3]))
            print("is closer? ", is_closer)
            print("closer rate:", np.sum(is_closer_list) / len(is_closer_list))
            rewards1 = []
            rewards2 = []
            observed_map, robot_pose = field.reset()
            player.reset()

            if (i_episode + 1) % 200 == 0:
                # plt.cla()
                model_save_path = os.path.join(params['model_folder'], "Agent_dqn_state_dict_%d.mdl" % (i_episode + 1))
                player.store_model(model_save_path)

print('Complete')
