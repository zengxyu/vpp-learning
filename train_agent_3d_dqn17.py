import sys
import os
import argparse

from scipy.spatial.transform.rotation import Rotation

from agent.agent_dqn import Agent
from field_env_3d_unknown_map import Field, Action
from memory.robot_pose_cluster import RobotPoseCluster
from network.network_dqn import DQN_Network5, DQN_Network6
from util.summary_writer import MySummaryWriter
from util.util import get_euclidean_distance

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import numpy as np

np.set_printoptions(precision=3)
"""
起始位置为随机
改了network,DQN_Network5
继续改reward

多放点信息到输入
考虑角度和方向的差异
现在action有13个了
考虑整个field: 能跑出来，效果不错

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
    'c_step': 20,
    'model': DQN_Network6,

    # train params
    'is_train': True,
    'visualise': True,
    'is_normalize': False,
    'num_episodes': 5000000,
    'scale': 15,
    'use_gpu': True,

    # folder params

    # output
    'output_folder': "output_dqn17",
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

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=500,
              init_file='VG07_6.binvox', headless=args.headless)
player = Agent(params, summary_writer)

robot_pose_cluster = RobotPoseCluster()
all_mean_rewards = []
all_mean_losses = []
time_step = 0

observed_map, robot_pose = field.reset()
distances_travelled = []
is_closer_list = []

record_pose_count = 0
update_freq = 5


for i_episode in range(params['num_episodes']):
    done = False
    # rewards = []
    rewards1 = []
    time_step = 0
    rewards2 = []

    destination = np.array([128, 128, 128])
    initial_direction = np.array([[-1], [0], [0]])
    init_observed_map, init_robot_pose = observed_map, robot_pose
    actions = []
    diff_directions = []
    destination_list = []
    while not done:
        destination = robot_pose_cluster.get_destination(robot_pose[:3])
        direction = destination - robot_pose[:3]

        # normalize direction
        unit_direction = direction / np.linalg.norm(direction)
        # robot direction
        robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ initial_direction

        # print(Rotation.from_quat(robot_pose[3:]).as_matrix().shape)
        # print(initial_direction.shape)
        # print(robot_direction.shape)
        # diff direction

        diff_direction = unit_direction - robot_direction.squeeze()
        diff_directions.append(int(np.linalg.norm(diff_direction) * 100) / 100)
        robot_pose_input = np.concatenate([direction, diff_direction, robot_pose], axis=0)
        action = player.act(observed_map, robot_pose_input)
        # if action <= 6:
        #     action = 1
        observed_map_next, robot_pose_next, reward1, reward3, done = field.step(action)

        # 这里构造奖励
        reward2 = 109 - get_euclidean_distance(robot_pose_next[:3], destination)
        reward2 = int(reward2)
        reward = reward2

        # 构造下一个状态
        direction_next = destination - robot_pose_next[:3]
        # normalize direction next
        unit_direction_next = direction_next / np.linalg.norm(direction_next)
        # robot direction next
        robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ initial_direction
        # diff direction next
        diff_direction_next = unit_direction_next - robot_direction_next.squeeze()

        robot_pose_input_next = np.concatenate([direction_next, diff_direction_next, robot_pose_next], axis=0)

        player.step(state=[observed_map, robot_pose_input], action=action, reward=reward,
                    next_state=[observed_map_next, robot_pose_input_next], done=done)

        # train
        if time_step % 2 == 0:
            loss = player.learn(memory_config_dir=params['memory_config_dir'])
            summary_writer.add_loss(loss)

        time_step += 1
        # record
        if reward1 > 20:
            robot_pose_cluster.add_robot_pose(robot_pose_next[:3])
        destination_list.append(destination.tolist())
        summary_writer.add_reward(reward1, i_episode)
        actions.append(action)
        rewards1.append(reward1)
        rewards2.append(reward2)
        # 转到下一个状态
        observed_map = observed_map_next.copy()
        robot_pose = robot_pose_next.copy()
        # done = done or get_euclidean_distance(robot_pose_next[:3], destination) == 0

        if not args.headless:
            threading.Thread.considerYield()

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
            print("rewards1:{}".format(rewards1))
            print("rewards2:{}".format(rewards2))
            print("actions:{}".format(actions))
            print("diff directions norm:{}".format(diff_directions))

            print("diff_direction_next:{}".format(diff_direction_next))

            # print("mean rewards2:{}; new visit cell num: {}".format(np.sum(rewards2), np.sum(rewards2) / r_ratio))
            is_closer = get_euclidean_distance(end_robot_pose[:3], destination) < get_euclidean_distance(
                init_robot_pose[:3], destination)
            is_closer_list.append(is_closer)
            print("in this episode, robot travels from {} to {}".format(init_robot_pose[:3], end_robot_pose[:3]))
            print("is closer? ", is_closer)
            print("closer rate:", np.sum(is_closer_list) / len(is_closer_list))
            print("destination_list:", np.array(destination_list))
            if i_episode % update_freq:
                robot_pose_cluster.update_cluster()
                record_pose_count = 0
            rewards1 = []
            rewards2 = []
            destination_list = []
            observed_map, robot_pose = field.reset()
            player.reset()

            if (i_episode + 1) % 200 == 0:
                # plt.cla()
                model_save_path = os.path.join(params['model_folder'], "Agent_dqn_state_dict_%d.mdl" % (i_episode + 1))
                player.store_model(model_save_path)

print('Complete')
