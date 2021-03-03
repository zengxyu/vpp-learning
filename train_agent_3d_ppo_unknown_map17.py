import os
from enum import IntEnum

import numpy as np
import argparse

from agent.agent_ppo_3d_unknown_map import Agent
from field_env_3d_unknown_map import Field, Action
from memory.GridCellAccessRecord import GridCellAccessRecord
from network.network_ppo_3d_unknown_map import PPOPolicy3DUnknownMap2, PPOPolicy3DUnknownMap4, PPOPolicy3DUnknownMap5
from util.summary_writer import MySummaryWriter
from util.util import get_euclidean_distance
"""
改了gamma
self.MOVE_STEP = 2
self.ROT_STEP = 45.0
改了初始位姿
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
    'lr': 1e-5,

    'model': PPOPolicy3DUnknownMap5,
    'output': "output_ppo_unknown_map17",
    'config_dir': "config_dir2"
}

if not os.path.exists(params['output']):
    os.mkdir(params['output'])

log_dir = os.path.join(params['output'], 'log')
summary_writer = MySummaryWriter(log_dir)

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=1000,
              init_file='VG07_6.binvox', headless=args.headless)
model_path = os.path.join(params['output'], "Agent_ppo_state_dict_99.mdl")

player = Agent(params, field, summary_writer, train_agent=True, normalize=True, model_path="")


# grid_cell_access_record = GridCellAccessRecord(shape=(256, 256, 256))


class Action(IntEnum):
    DO_NOTHING = 0,
    MOVE_FORWARD = 1,
    MOVE_BACKWARD = 2,
    MOVE_LEFT = 3,
    MOVE_RIGHT = 4,
    MOVE_UP = 5,
    MOVE_DOWN = 6,
    ROTATE_ROLL_P = 7,
    ROTATE_ROLL_N = 8,
    ROTATE_PITCH_P = 9,
    ROTATE_PITCH_N = 10,
    ROTATE_YAW_P = 11,
    ROTATE_YAW_N = 12

def main_loop():
    global field, args
    episodes = 200000

    distances_travelled = []
    destination = [128, 128, 128]
    is_closer_list = []
    for i in range(0, episodes):
        done = False
        ts = 0
        rewards = []
        rewards1 = []
        rewards2 = []
        penalty = -1
        pre_distance = 0
        observed_map, robot_pose = field.reset()

        init_observed_map, init_robot_pose = observed_map, robot_pose

        while not done:

            action = player.get_action(observed_map, robot_pose)
            observed_map_prime, robot_pose_prime, reward1, reward3, done = field.step(action)
            # if np.sum(robot_pose[:3] - robot_pose_prime[:3]) != 0:
            #     print("robot pose:{}".format(robot_pose))
            #     print("robot pose prime:{}".format(robot_pose_prime))
            r_ratio = 5
            # reward2 = grid_cell_access_record.get_reward_of_new_visit(robot_pose_prime) * r_ratio
            # reward = reward1 + reward3
            # penalty = penalty * 1.01 if reward1 == 0 else penalty * 0.99
            # penalty = 0 if reward1 > 0 else -3

            reward2 = -10
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
            elif get_euclidean_distance(robot_pose_prime[:3], destination) < 130:
                reward2 = -1
            elif get_euclidean_distance(robot_pose_prime[:3], destination) < 150:
                reward2 = -3
            elif get_euclidean_distance(robot_pose_prime[:3], destination) < 170:
                reward2 = -5
            elif get_euclidean_distance(robot_pose_prime[:3], destination) < 180:
                reward2 = -7
            # current_distance = np.sqrt(np.sum(np.square(robot_pose_prime[:3] - init_robot_pose[:3])))
            # reward2 = current_distance - pre_distance
            # print("\npre_distance:", pre_distance)
            # print("current_distance:", current_distance)
            # print("reward2:", reward2)
            # pre_distance = current_distance

            reward = reward2

            rewards1.append(reward1)
            rewards2.append(reward2)
            rewards.append(reward)

            player.store_reward(reward, done)
            summary_writer.add_reward(reward1, i)

            observed_map = observed_map_prime.copy()
            robot_pose = robot_pose_prime.copy()

            if not args.headless:
                threading.Thread.considerYield()

            ts += 1

            if done:
                player.reset()
                end_observed_map, end_robot_pose = observed_map, robot_pose
                distance_travelled = get_euclidean_distance(end_robot_pose[:3], init_robot_pose[:3])
                distances_travelled.append(distance_travelled)
                # grid_cell_access_record.clear()
                summary_writer.add_episode_len(ts, i)
                summary_writer.add_distance(get_euclidean_distance(end_robot_pose[:3], destination), i)
                print("\nepisode {} over".format(i))
                print("mean rewards1:{}".format(np.sum(rewards1)))
                print("mean rewards2:{}".format(np.sum(rewards2)))

                print("mean rewards:{}".format(np.sum(rewards)))
                print("rewards2:{}".format(rewards2))

                print("time steps:{}".format(ts))
                print("learning rate:{}".format(player.optimizer.param_groups[0]['lr']))
                print("distance travelled:{}".format(distance_travelled))
                print("max distance travelled:{}".format(np.max(distances_travelled)))

                print("distance travelled:{}".format(distance_travelled))
                print("max distance travelled:{}".format(np.max(distances_travelled)))
                print("in this episode, robot travels from {} to {}".format(init_robot_pose[:3], end_robot_pose[:3]))
                is_closer = get_euclidean_distance(end_robot_pose[:3], destination) < get_euclidean_distance(
                          init_robot_pose[:3],
                          destination)
                is_closer_list.append(is_closer)
                print("is closer? ", is_closer)
                print("closer rate:", np.sum(is_closer_list) / len(is_closer_list))
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
