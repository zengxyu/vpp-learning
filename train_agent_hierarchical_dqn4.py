import sys
import os
import argparse

from scipy.spatial.transform.rotation import Rotation

from agent.agent_hierarchical_dqn import Agent
from field_env_3d_unknown_map import Field, Action
from memory.robot_pose_cluster import RobotPoseCluster
from network.network_dqn import DQN_Network5, DQN_Network6
from network.network_hierarchical_dqn import NetworkManager
from util.summary_writer import MySummaryWriter
from util.util import get_eu_distance

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import numpy as np

np.set_printoptions(precision=2)
"""

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
    'manager_update_every': 20,
    'manager_model': NetworkManager,
    'eps_start': 0.15,  # Default/starting value of eps
    'eps_decay': 0.99999,  # Epsilon decay rate
    'eps_min': 0.15,  # Minimum epsilon
    'gamma': 0.9,
    'buffer_size': 200000,
    'batch_size': 16,
    'action_size': len(Action),

    'is_double': False,
    'is_priority_buffer': True,

    # grid params
    'max_step': 500,
    'c_freq': 20,
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
    'output_folder': "output_worker4",
    'log_folder': 'log',
    'model_folder': 'model',
    'memory_config_dir': "memory_config",

    'eps': 1e-15,
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

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=params['max_step'],
              init_file='VG07_6.binvox', headless=args.headless)
player = Agent(params, summary_writer)

all_mean_rewards = []
all_mean_losses = []
time_step = 0

distances_travelled = []
is_closer_list = []

record_pose_count = 0
update_freq = 5


class WorkerTrainer:
    def __init__(self):
        self.initial_direction = np.array([[-1], [0], [0]])
        self.time_step = 0
        self.thresh = 50
        self.max_step = 1.5 * self.thresh

        self.worker_dict = self.reset_worker_dict()
        self.is_closer_list = []

    def train_worker(self, num_episode):
        """
        The input of the worker : goal(direction)
        """

        for i_episode in range(num_episode):
            done = False

            observed_map, robot_pose = field.reset()

            pose_destination = self.get_robot_pose_delta(robot_pose)
            self.worker_dict = self.reset_worker_dict()

            self.time_step = 0

            while not done:
                # rotation
                # robot_pose_input = np.concatenate([pose_delta, rotation_delta, robot_pose], axis=0)

                robot_pose_input = self.get_robot_pose_input(pose_destination, robot_pose)
                # print("robot pose input:", robot_pose_input)
                action = player.act(robot_pose_input)

                observed_map_next, robot_pose_next, ext_reward1, ext_reward2, done = field.step(action)

                # 这里构造奖励
                intrinsic_pose_reward = 109 - get_eu_distance(robot_pose_next[:3], pose_destination)
                # intrinsic_rotation_reward = - np.linalg.norm(rotation_delta)

                worker_reward = intrinsic_pose_reward
                # print("worker_reward:", worker_reward)

                # self.store_info(worker_reward, robot_pose, action, pose_delta)
                self.store_info(worker_reward, robot_pose, action, 0)

                # 构造下一个状态
                # pose_delta = robot_pose[:3] + pose_delta - robot_pose_next[:3]

                # rotation_delta = self.trans_quat_to_direction(
                #     robot_pose[3:]) + rotation_delta - self.trans_quat_to_direction(robot_pose_next[3:])
                # rotation_delta = np.zeros((3,))

                # robot_pose_input_next = np.concatenate([pose_delta, rotation_delta, robot_pose_next], axis=0)
                robot_pose_input_next = self.get_robot_pose_next_input(pose_destination, robot_pose_next)
                # print("robot_pose_input_next:", robot_pose_input_next)

                player.store_worker_experience(state=[observed_map, robot_pose_input], action=action,
                                               reward=worker_reward,
                                               next_state=[observed_map_next, robot_pose_input_next], done=done)

                # 转到下一个状态
                observed_map = observed_map_next.copy()
                robot_pose = robot_pose_next.copy()

                done = done or self.time_step >= self.max_step

                self.time_step += 1

                if not args.headless:
                    threading.Thread.considerYield()

                if (i_episode + 1) % 200 == 0:
                    # plt.cla()
                    model_save_path = os.path.join(params['model_folder'],
                                                   "Agent_dqn_state_dict_%d.mdl" % (i_episode + 1))
                    player.store_model(model_save_path)
                # train
                if self.time_step % 2 == 0:
                    loss = player.learn_worker()
                if done:
                    print("destination:{}".format(pose_destination))
                    self.print_info()
                    print("\nepisode {} over".format(i_episode))

    def get_robot_pose_next_input(self, pose_destination, robot_pose_next):
        # 构造下一个状态
        direction_next = pose_destination - robot_pose_next[:3]
        # normalize direction next
        unit_direction_next = direction_next / np.linalg.norm(direction_next)
        # robot direction next
        robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ self.initial_direction
        # diff direction next
        diff_direction_next = unit_direction_next - robot_direction_next.squeeze()

        robot_pose_input_next = np.concatenate([direction_next, diff_direction_next, robot_pose_next], axis=0)
        return robot_pose_input_next

    def get_robot_pose_delta(self, robot_pose):
        pose_delta = np.random.randint(0, self.thresh, (3,))

        # rotation_delta = np.random.random((3,))
        rotation_delta = np.zeros((3,))
        destination = np.clip(pose_delta + robot_pose[:3], 0, 255)
        pose_destination = destination[:3]
        rotation_destination = destination[3:]
        return pose_destination

    def get_robot_pose_input(self, pose_destination, robot_pose):
        # rotation_delta = rotation_delta / np.linalg.norm(rotation_delta) + params['eps']

        pose_direction = pose_destination - robot_pose[:3]

        # rotation
        # normalize direction
        unit_direction = pose_direction / np.linalg.norm(pose_direction)
        # robot direction
        robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ self.initial_direction

        diff_direction = unit_direction - robot_direction.squeeze()
        robot_pose_input = np.concatenate([pose_direction, diff_direction, robot_pose], axis=0)
        return robot_pose_input

    def trans_quat_to_direction(self, cur_rotation_quat):
        # print("cur_rotation_quat:", cur_rotation_quat)
        cur_direction = Rotation.from_quat(cur_rotation_quat).as_matrix() @ self.initial_direction
        cur_direction = cur_direction.squeeze()
        # print("cur_direction:", cur_direction)
        return cur_direction

    def store_info(self, worker_reward, robot_pose, action, pose_delta):
        self.worker_dict['worker_rewards'].append(worker_reward)
        self.worker_dict['robot_poses'].append(robot_pose.tolist())
        self.worker_dict['actions'].append(action)
        self.worker_dict['pose_deltas'].append(pose_delta)

    def print_info(self):
        print("max step:{}".format(self.max_step))
        print("worker_rewards:{}".format(np.array(self.worker_dict['worker_rewards'])))
        # print("robot_poses:{}".format(np.array(self.worker_dict['robot_poses'])))
        # print("pose_deltas:{}".format(np.array(self.worker_dict['pose_deltas'])))
        print("actions:{}".format(self.worker_dict['actions']))

        start_position = self.worker_dict['robot_poses'][0][:3]
        end_position = self.worker_dict['robot_poses'][-1][:3]
        start_delta = self.worker_dict['pose_deltas'][0]
        end_delta = self.worker_dict['pose_deltas'][-1]

        destination = np.array(start_position) + start_delta
        # print("start_delta : {}".format(start_delta))
        # print("end_delta : {}".format(end_delta))
        # print("destination:{}".format(destination))
        print("start position, end_position : {},{}".format(start_position, end_position))
        print("distance_traveled : {}".format(get_eu_distance(start_position, end_position)))

        is_closer = get_eu_distance(end_position, destination) < get_eu_distance(start_position, destination)
        self.is_closer_list.append(is_closer)

        print("is closer? ", is_closer)
        print("closer rate:", np.sum(self.is_closer_list) / len(self.is_closer_list))

    def reset_worker_dict(self):
        return {"worker_rewards": [], "robot_poses": [], "actions": [], "pose_deltas": []}


class ManagerTrainer:
    def __init__(self):
        pass

    def train_manager(self):
        pass

    def train(self):

        for i_episode in range(params['num_episodes']):
            done = False
            # rewards = []
            rewards1 = []
            time_step = 0
            rewards2 = []

            initial_direction = np.array([[-1], [0], [0]])
            init_observed_map, init_robot_pose = observed_map, robot_pose
            goal = player.manager_act(observed_map, robot_pose)

            actions = []
            diff_directions = []
            destination_list = []

            time_step_episode = 0
            manager_reward = 0
            manager_observed_map = None
            manager_robot_pose = None
            manager_rewards = []
            while not done:
                # get the destination, the destination should be trans to a standard form
                destination = np.clip(goal, 0, 255)
                pose_destination = destination[:3]
                rotation_destination = destination[3:]

                pose_direction = pose_destination - robot_pose[:3]

                # rotation
                # normalize direction
                unit_direction = pose_direction / np.linalg.norm(pose_direction)
                # robot direction
                robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ initial_direction

                diff_direction = unit_direction - robot_direction.squeeze()
                diff_directions.append(int(np.linalg.norm(diff_direction) * 100) / 100)
                robot_pose_input = np.concatenate([pose_direction, diff_direction, robot_pose], axis=0)
                action = player.act(robot_pose_input)

                observed_map_next, robot_pose_next, reward1, reward3, done = field.step(action)

                # 这里构造奖励
                reward2 = 109 - get_eu_distance(robot_pose_next[:3], pose_destination)
                reward2 = int(reward2)
                reward = reward2
                manager_reward += reward1

                # 构造下一个状态
                direction_next = pose_destination - robot_pose_next[:3]
                # normalize direction next
                unit_direction_next = direction_next / np.linalg.norm(direction_next)
                # robot direction next
                robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ initial_direction
                # diff direction next
                diff_direction_next = unit_direction_next - robot_direction_next.squeeze()

                robot_pose_input_next = np.concatenate([direction_next, diff_direction_next, robot_pose_next], axis=0)

                player.store_worker_experience(state=[observed_map, robot_pose_input], action=action, reward=reward,
                                               next_state=[observed_map_next, robot_pose_input_next], done=done)
                if time_step_episode % params['c_freq'] == 0:
                    manager_observed_map, manager_robot_pose = observed_map, robot_pose

                if (time_step_episode + 1) % params['c_freq'] == 0:
                    manager_rewards.append(manager_reward)
                    destination_list.append(destination.tolist())
                    player.store_manager_experience(state=[manager_observed_map, manager_robot_pose, np.array(goal)],
                                                    reward=manager_reward,
                                                    next_state=[observed_map_next, robot_pose_next], done=done)
                    goal = player.manager_act(observed_map, robot_pose)

                    manager_reward = 0
                # train
                if time_step % 2 == 0:
                    loss = player.learn_worker()
                    # print("loss  ---  learn worker")
                    summary_writer.add_loss(loss)

                if time_step % (2 * params['c_freq']) == 0:
                    loss = player.learn_manager()

                time_step += 1
                time_step_episode += 1
                # record
                # destination_list.append(destination.tolist())
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
                    distance_travelled = get_eu_distance(end_robot_pose[:3], init_robot_pose[:3])
                    distances_travelled.append(distance_travelled)
                    print("\nepisode {} over".format(i_episode))
                    print("mean rewards1:{}".format(np.sum(rewards1)))
                    print("mean rewards2:{}".format(np.sum(rewards2)))
                    print("distance travelled:{}".format(distance_travelled))
                    print("max distance travelled:{}".format(np.max(distances_travelled)))
                    print("rewards1:{}".format(rewards1))
                    print("rewards2:{}".format(rewards2))
                    print("manager rewards:{}".format(manager_rewards))

                    print("actions:{}".format(actions))
                    print("diff directions norm:{}".format(diff_directions))

                    print("diff_direction_next:{}".format(diff_direction_next))

                    # print("mean rewards2:{}; new visit cell num: {}".format(np.sum(rewards2), np.sum(rewards2) / r_ratio))
                    is_closer = get_eu_distance(end_robot_pose[:3], pose_destination) < get_eu_distance(
                        init_robot_pose[:3], pose_destination)
                    is_closer_list.append(is_closer)
                    print(
                        "in this episode, robot travels from {} to {}".format(init_robot_pose[:3], end_robot_pose[:3]))
                    print("is closer? ", is_closer)
                    print("closer rate:", np.sum(is_closer_list) / len(is_closer_list))
                    print("destination_list:", np.array(destination_list))
                    rewards1 = []
                    rewards2 = []
                    destination_list = []
                    observed_map, robot_pose = field.reset()
                    player.reset()

                    if (i_episode + 1) % 200 == 0:
                        # plt.cla()
                        model_save_path = os.path.join(params['model_folder'],
                                                       "Agent_dqn_state_dict_%d.mdl" % (i_episode + 1))
                        player.store_model(model_save_path)

        print('Complete')


if __name__ == '__main__':
    worker_trainer = WorkerTrainer()
    worker_trainer.train_worker(num_episode=10000)
