import os

from agent.agent_my_ppo2 import Agent
from field_env_3d_unknown_map import Field, Action
from memory.robot_pose_cluster import RobotPoseCluster
from network.network_ppo import PPO
from network.network_ppo_3d_unknown_map import PPOPolicy3DUnknownMap2, PPOPolicy3DUnknownMap3, PPOPolicy3DUnknownMap4
from util.summary_writer import MySummaryWriter
from util.util import get_euclidean_distance

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pygame
import pygame as pg
import torch
from torch.utils.tensorboard import SummaryWriter

"""
change the ste size to 10
"""
parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=True, action="store_true", help="Run in headless mode")
args = parser.parse_args()

if not args.headless:
    from direct.stdpy import threading

params = {
    "num_episodes": 1000000,
    'gamma': 0.98,
    'lr': 1e-4,

    'batch_size': 128,
    'seq_len': 4,
    'model': PPOPolicy3DUnknownMap4,  # PPO_LSTM, PPO_LSTM2, PPO_LSTM3
    'action_size': len(Action),
    'robot_pose_size': 13,
    'output': "output_my_ppo_09",
    'config_dir': "config_dir",

    "use_gpu": True,
}

if not os.path.join(params['output']):
    os.mkdir(params['output'])

log_dir = os.path.join(params['output'], 'log')
summary_writer = MySummaryWriter(log_dir)

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=1000,
              init_file='VG07_6.binvox', headless=args.headless)

player = Agent(params, summary_writer, normalize=True)
robot_pose_cluster = RobotPoseCluster(k=5, field_size=(256, 256, 256), verbose=True)

total_rewards, smoothed_rewards = [], []

global_step = 0
T_horizon = params['seq_len']
batch_size = params["batch_size"]
all_mean_rewards = []
all_mean_losses = []
distances_travelled = []
is_closer_list = []

for i in range(0, params['num_episodes']):
    print("\n --- \n \n Episode:{}".format(i))

    done = False
    ts = 0

    rewards = []
    rewards1 = []
    rewards2 = []
    rewards3 = []

    losses = []

    observed_map, robot_pose = field.reset()

    init_observed_map, init_robot_pose = observed_map, robot_pose

    robot_pose_cluster.add_robot_pose(init_robot_pose[:3])
    destination = robot_pose_cluster.get_destination(init_robot_pose[:3])
    previous_distance_to_destination = get_euclidean_distance(init_robot_pose[:3], destination)

    while not done:
        global_step += 1
        robot_pose_combined = np.concatenate([init_robot_pose[:3], destination, robot_pose])
        action, value, probs = player.act(observed_map, robot_pose_combined)

        observed_map_prime, robot_pose_prime, reward1, reward3, done = field.step(action.detach().cpu().numpy()[0])
        robot_pose_combined_prime = np.concatenate([init_robot_pose[:3], destination, robot_pose_prime])

        reward2 = 0
        if get_euclidean_distance(robot_pose_prime[:3], destination) == 0:
            reward2 = 100
        elif get_euclidean_distance(robot_pose_prime[:3], destination) < 5:
            reward2 = 5

        if get_euclidean_distance(robot_pose[:3],robot_pose_prime[:3]) ==0:
            reward2 = -1
        # elif get_euclidean_distance(robot_pose_prime[:3], destination) < 30:
        #     reward2 = 5
        # elif get_euclidean_distance(robot_pose_prime[:3], destination) < 50:
        #     reward2 = 4
        # elif get_euclidean_distance(robot_pose_prime[:3], destination) < 70:
        #     reward2 = 3
        # elif get_euclidean_distance(robot_pose_prime[:3], destination) < 90:
        #     reward2 = 2
        # elif get_euclidean_distance(robot_pose_prime[:3], destination) < 110:
        #     reward2 = 1
        # elif get_euclidean_distance(robot_pose_prime[:3], destination) < 130:
        #     reward2 = -1
        # elif get_euclidean_distance(robot_pose_prime[:3], destination) < 150:
        #     reward2 = -2
        # elif get_euclidean_distance(robot_pose_prime[:3], destination) < 170:
        #     reward2 = -3
        # elif get_euclidean_distance(robot_pose_prime[:3], destination) < 180:
        #     reward2 = -4
        reward = reward2

        rewards1.append(reward1)
        rewards2.append(reward2)
        # rewards3.append(reward3)
        rewards.append(reward)

        player.store_data(
            [observed_map, robot_pose_combined, action.detach().cpu().numpy(), reward, observed_map_prime,
             robot_pose_combined_prime, value.detach().cpu().numpy().squeeze(), probs.detach().cpu().numpy().squeeze(),
             done])

        observed_map = observed_map_prime.copy()
        robot_pose = robot_pose_prime.copy()

        summary_writer.add_reward(reward1, i)
        ts += 1
        if not args.headless:
            threading.Thread.considerYield()

        if done:
            end_observed_map, end_robot_pose = observed_map, robot_pose
            distance_travelled = get_euclidean_distance(end_robot_pose[:3], init_robot_pose[:3])
            distances_travelled.append(distance_travelled)
            # grid_cell_access_record.clear()
            summary_writer.add_episode_len(ts, i)

            # robot_pose_cluster.update_cluster()
            print("\nepisode {} over".format(i))
            print("mean rewards1:{}".format(np.sum(rewards1)))
            print("mean rewards2:{}".format(np.sum(rewards2)))
            print("mean rewards:{}".format(np.sum(rewards)))
            print("rewards2:{}".format(rewards2))

            print("time steps:{}".format(ts))
            print("learning rate:{}".format(player.optimizer.param_groups[0]['lr']))
            print("distance travelled:{}".format(distance_travelled))
            print("max distance travelled:{}".format(np.max(distances_travelled)))
            print("in this episode, robot travels from {} to {}".format(init_robot_pose[:3], end_robot_pose[:3]))
            print("is closer? ",
                  get_euclidean_distance(end_robot_pose[:3], destination) < get_euclidean_distance(init_robot_pose[:3],
                                                                                                   destination))
            # print("mean rewards2:{}; new visit cell num: {}".format(np.sum(rewards2), np.sum(rewards2) / r_ratio))
            is_closer = get_euclidean_distance(end_robot_pose[:3], destination) < get_euclidean_distance(
                init_robot_pose[:3],
                destination)
            is_closer_list.append(is_closer)
            print("is closer? ", is_closer)
            print("closer rate:", np.sum(is_closer_list) / len(is_closer_list))
            break

        if player.memory.is_full_batch():
            # print("train net")
            player.train_net()
            player.memory.reset_data()

if not args.headless:
    pg.quit()
