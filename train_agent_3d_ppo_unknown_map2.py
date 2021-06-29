import os
from enum import IntEnum

import numpy as np
import argparse

from scipy.spatial.transform.rotation import Rotation

from old_agent.agent_ppo_3d_unknown_map import Agent
from field_env_3d_unknown_map2 import Field, Action
from network.network_ppo_3d_unknown_map import PPOPolicy3DUnknownMap6
from util.summary_writer import MySummaryWriter
from util.util import get_eu_distance

"""
robot_pose_input includes  np.concatenate([direction, diff_direction, robot_pose], axis=0)

"""
parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=True, action="store_true", help="Run in headless mode")
args = parser.parse_args()

if not args.headless:
    from direct.stdpy import threading

params = {
    'action': Action,

    'traj_collection_num': 128,
    'traj_len': 4,
    'gamma': 0.98,
    'lr': 1e-4,

    'model': PPOPolicy3DUnknownMap6,
    'output': "output_ppo_unknown_map2",
    'config_dir': "config_dir2"
}

if not os.path.exists(params['output']):
    os.mkdir(params['output'])

log_dir = os.path.join(params['output'], 'log')
summary_writer = MySummaryWriter(log_dir)

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=500,
              init_file='VG07_6.binvox', headless=args.headless)
model_path = os.path.join(params['output'], "Agent_ppo_state_dict_99.mdl")

player = Agent(params, field, summary_writer, train_agent=True, normalize=False, model_path="")


def main_loop():
    global field, args
    episodes = 200000

    initial_direction = np.array([[1], [0], [0]])

    distances_travelled = []
    destination = np.array([128, 128, 128])
    is_closer_list = []

    for i in range(0, episodes):
        done = False
        ts = 0
        rewards = []
        rewards1 = []
        rewards2 = []
        observed_map, robot_pose = field.reset()

        init_observed_map, init_robot_pose = observed_map, robot_pose
        actions = []
        diff_directions = []

        while not done:
            # robot direction
            robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ initial_direction
            robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)

            action = player.get_action(observed_map, robot_pose_input)
            action = action.cpu().numpy()[0]

            observed_map_prime, robot_pose_prime, reward1, reward3, done = field.step(action)


            # here to record
            ts += 1

            rewards1.append(reward1)
            actions.append(action)

            player.store_reward(reward1, done)
            summary_writer.add_reward(reward1, i)

            observed_map = observed_map_prime.copy()
            robot_pose = robot_pose_prime.copy()

            if not args.headless:
                threading.Thread.considerYield()

            if done:
                player.reset()
                end_observed_map, end_robot_pose = observed_map, robot_pose
                distance_travelled = get_eu_distance(end_robot_pose[:3], init_robot_pose[:3])
                distances_travelled.append(distance_travelled)
                # grid_cell_access_record.clear()
                summary_writer.add_episode_len(ts, i)
                summary_writer.add_distance(get_eu_distance(end_robot_pose[:3], destination), i)
                print("\nepisode {} over".format(i))
                print("mean rewards1:{}".format(np.sum(rewards1)))
                print("mean rewards:{}".format(np.sum(rewards)))
                print("rewards1:{}".format(rewards1))
                print("actions:{}".format(actions))
                print("time steps:{}".format(ts))
                print("learning rate:{}".format(player.optimizer.param_groups[0]['lr']))
                # print("distance travelled:{}".format(distance_travelled))
                # print("max distance travelled:{}".format(np.max(distances_travelled)))
                #
                # print("distance travelled:{}".format(distance_travelled))
                # print("max distance travelled:{}".format(np.max(distances_travelled)))
                print("in this episode, robot travels from {} to {}".format(init_robot_pose[:3], end_robot_pose[:3]))
                # is_closer = get_eu_distance(end_robot_pose[:3], destination) < get_eu_distance(
                #     init_robot_pose[:3],
                #     destination)
                # is_closer_list.append(is_closer)
                # print("is closer? ", is_closer)
                # print("closer rate:", np.sum(is_closer_list) / len(is_closer_list))
                # print("mean rewards2:{}; new visit cell num: {}".format(np.sum(rewards2), np.sum(rewards2) / r_ratio))

                rewards1 = []
                rewards2 = []

        if (i + 1) % 100 == 0:
            player.store_model(os.path.join(params['output'], "Agent_ppo_state_dict_{}.mdl".format(i)))


if args.headless:
    main_loop()
else:
    # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
    # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
    main_thread = threading.Thread(target=main_loop)
    main_thread.start()
    field.gui.run()
