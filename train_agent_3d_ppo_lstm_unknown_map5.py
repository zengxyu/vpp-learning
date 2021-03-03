import os
import argparse

import torch
import numpy as np

from agent.agent_ppo_lstm_3d_unknown_map import Agent
from field_env_3d_unknown_map import Field, Action
from network.network_ppo_lstm import PPO_LSTM2, PPO_LSTM3, PPO_LSTM4, PPO_LSTM5
from util.summary_writer import MySummaryWriter
from util.util import get_euclidean_distance

parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=True, action="store_true", help="Run in headless mode")
args = parser.parse_args()

if not args.headless:
    from direct.stdpy import threading

params = {
    'traj_collection_num': 16,
    'traj_len': 4,
    'gamma': 0.98,
    'lr': 1e-5,

    'robot_pose_size': 12,

    'batch_size': 16,
    'seq_len': 4,
    'num_layers': 4,
    'model': PPO_LSTM5,  # PPO_LSTM, PPO_LSTM2, PPO_LSTM3
    'action_size': len(Action),
    'output': "output_ppo_lstm_unknown_map5",
    'config_dir': "config_dir"

}

if not os.path.join(params['output']):
    os.mkdir(params['output'])

log_dir = os.path.join(params['output'], 'log')
summary_writer = MySummaryWriter(log_dir)

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=1000,
              init_file='VG07_6.binvox', headless=args.headless)
player = Agent(params, summary_writer, normalize=False)


def main_loop():
    global field, args
    episodes = 200000
    distances_travelled = []
    destination = [128, 128, 128]
    for i in range(0, episodes):
        observed_map, robot_pose = field.reset()
        init_observed_map, init_robot_pose = observed_map, robot_pose
        pre_distance = 0

        h_out = torch.zeros([params['num_layers'], 1, 32], dtype=torch.float)
        c_out = torch.zeros([params['num_layers'], 1, 32], dtype=torch.float)

        rewards = []
        rewards1 = []
        rewards2 = []

        done = False
        while not done:
            h_in, c_in = h_out, c_out
            action, value, probs, h_out, c_out = player.act(observed_map, robot_pose, h_in, c_in)
            observed_map_prime, robot_pose_prime, reward1, reward3, done = field.step(
                action.detach().cpu().numpy()[0][0])

            current_distance = np.sqrt(np.sum(np.square(robot_pose_prime[:3] - init_robot_pose[:3])))
            # reward2 = np.log10(current_distance + 1)
            # ratio = 5
            # reward2 = (np.exp(current_distance - pre_distance) - 1) * ratio

            # print("\npre_distance:", pre_distance)
            # print("current_distance:", current_distance)
            # print("reward2:", reward3)
            # pre_distance = current_distance

            # reward = reward1 + reward2

            reward2 = 0
            if get_euclidean_distance(robot_pose_prime[:3], destination) < 10:
                reward2 = 15
            elif get_euclidean_distance(robot_pose_prime[:3], destination) < 20:
                reward2 = 11
            elif get_euclidean_distance(robot_pose_prime[:3], destination) < 30:
                reward2 = 9
            elif get_euclidean_distance(robot_pose_prime[:3], destination) < 50:
                reward2 = 7
            elif get_euclidean_distance(robot_pose_prime[:3], destination) < 70:
                reward2 = 5
            elif get_euclidean_distance(robot_pose_prime[:3], destination) < 90:
                reward2 = 3
            elif get_euclidean_distance(robot_pose_prime[:3], destination) < 110:
                reward2 = 1
            reward = reward2

            player.store_data(
                [observed_map, robot_pose, action.detach().cpu().numpy().squeeze(), reward, observed_map_prime,
                 robot_pose_prime, value.detach().cpu().numpy().squeeze(), probs.detach().cpu().numpy().squeeze(),
                 done, h_in.detach().cpu().numpy(), c_in.detach().cpu().numpy(), h_out.detach().cpu().numpy(),
                 c_out.detach().cpu().numpy()])

            observed_map = observed_map_prime.copy()
            robot_pose = robot_pose_prime.copy()

            summary_writer.add_reward(reward1, i)

            rewards1.append(reward1)
            rewards2.append(reward2)
            rewards.append(reward)

            if not args.headless:
                threading.Thread.considerYield()

            if done:
                end_observed_map, end_robot_pose = observed_map, robot_pose
                distance_travelled = np.sqrt(np.sum(np.square(end_robot_pose[:3] - init_robot_pose[:3])))
                distances_travelled.append(distance_travelled)

                print("\nepisode {} over".format(i))
                print("mean rewards1:{}".format(np.sum(rewards1)))
                print("mean rewards2:{}".format(np.sum(rewards2)))
                print("mean rewards:{}".format(np.sum(rewards)))
                print("init_robot_pose:{}; end_robot_pose:{}; ".format(init_robot_pose[:3], end_robot_pose[:3]))
                print("distance travelled:{}".format(distance_travelled))
                print("max distance travelled:{}".format(np.max(distances_travelled)))
                print("in this episode, robot travels from {} to {}".format(init_robot_pose[:3], end_robot_pose[:3]))
                print("is closer? ",
                      get_euclidean_distance(end_robot_pose[:3], destination) < get_euclidean_distance(
                          init_robot_pose[:3],
                          destination))
            if player.memory.is_full_batch():
                loss = player.train_net()

                player.memory.reset_data()

                summary_writer.add_loss(loss)


if args.headless:
    main_loop()
else:
    main_thread = threading.Thread(target=main_loop)
    main_thread.start()
    field.gui.run()
