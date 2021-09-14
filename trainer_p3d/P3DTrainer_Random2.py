import math
import os.path
import pickle
import random

import numpy as np
import time
from scipy.spatial.transform.rotation import Rotation

from utilities.StateDeque import Pose_State_DEQUE
from utilities.basic_logger import BasicLogger
from utilities.summary_writer import SummaryWriterLogger
from math import *

headless = True
if not headless:
    from direct.stdpy import threading


class P3DTrainer(object):
    def __init__(self, config, agent, field):
        self.config = config
        self.summary_writer = SummaryWriterLogger(config)

        self.logger = BasicLogger.setup_console_logging(config)
        self.agent = agent
        self.field = field
        self.seq_len = 5
        self.max_steps = self.field.max_steps
        self.deque = None
        self.time_step = 0
        self.initial_direction = np.array([[1], [0], [0]])
        self.last_targets_found = 0

        self.rewards_sum_n_episodes = []
        self.found_targets_sum_n_episodes = []

    def train(self, is_randomize, is_reward_plus_unknown_cells, randomize_control, randomize_from_48_envs, seq_len,
              is_save_path, is_stop_n_zero_rewards, is_map_diff_reward, is_add_negative_reward, is_save_env):
        self.seq_len = seq_len
        self.deque = Pose_State_DEQUE(capacity=self.seq_len)

        if headless:
            self.main_loop(is_randomize, is_reward_plus_unknown_cells, randomize_control, randomize_from_48_envs,
                           is_save_path, is_stop_n_zero_rewards, is_map_diff_reward, is_add_negative_reward,
                           is_save_env)
        else:
            # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
            # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
            main_thread = threading.Thread(target=self.main_loop)
            main_thread.start()
            self.field.gui.run()

    def main_loop(self, is_randomize, is_reward_plus_unknown_cells, randomize_control, randomize_from_48_envs,
                  is_save_path, is_stop_n_zero_rewards, is_map_diff_reward, is_add_negative_reward, is_save_env):
        diagonal_path_actions = []

        paths = []
        for i_episode in range(self.config.num_episodes_to_run):
            print("\nepisode {}".format(i_episode))
            e_start_time = time.time()
            done = False
            losses = []
            rewards = []
            found_targets = []
            unknown_cells = []
            known_cells = []
            actions = []
            path = []
            step = 0
            zero_found_target_consistent_count = 0

            self.agent.reset()

            _, observed_map, robot_pose, _ = self.field.reset(is_randomize=is_randomize,
                                                              randomize_control=randomize_control,
                                                              randomize_from_48_envs=randomize_from_48_envs,
                                                              is_save_env=is_save_env,
                                                              last_targets_found=self.last_targets_found)
            print("robot pose:{}".format(robot_pose))
            print("observation size:{}; robot pose size:{}".format(observed_map.shape, robot_pose.shape))
            while not done:
                loss = 0

                # robot direction
                robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ self.initial_direction
                robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)
                self.deque.append(robot_pose_input)
                # print(step, robot_pose)
                # 前5步，随便选择一个动作
                action = random.randint(0, self.field.get_action_size() - 1)
                # action = diagonal_path_actions[step]
                (_, observed_map_next,
                 robot_pose_next), found_target_num, unknown_cells_num, known_cells_num, _, _, done, _ = self.field.step(
                    action)

                # a = found_target_num
                # b = 0.05 * unknown_cells_num ** (
                #         1 - step / self.max_steps / 2)
                # c = - (known_cells_num / 2000) ** 1.1
                reward = found_target_num

                # 奖励unknown
                if is_reward_plus_unknown_cells:
                    reward += 0.006 * unknown_cells_num

                # if robot_pose is the same with the robot_pose_next, then reward--
                robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ self.initial_direction

                # diff direction
                robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)

                self.deque.append_next(robot_pose_input_next)
                path.append(robot_pose)
                # to the next state
                observed_map = observed_map_next.copy()
                robot_pose = robot_pose_next.copy()

                actions.append(action)
                rewards.append(reward)
                found_targets.append(found_target_num)
                unknown_cells.append(unknown_cells_num)
                known_cells.append(known_cells_num)
                losses.append(loss)
                self.time_step += 1
                step += 1
                # record
                if not headless:
                    threading.Thread.considerYield()

                if done:
                    step = 0
                    self.deque.clear()
                    self.last_targets_found = np.sum(found_targets)

                    self.print_info(i_episode, robot_pose, actions, rewards, found_targets, unknown_cells, known_cells,
                                    losses)
                    self.rewards_sum_n_episodes.append(np.sum(rewards))
                    self.found_targets_sum_n_episodes.append(np.sum(found_targets))
                    if is_save_path:
                        file_path = os.path.join(self.config.folder["out_folder"], "path_{}.obj".format(i_episode + 1))
                        pickle.dump(path, open(file_path, "wb"))
                        print("save robot path to file_path:{}".format(file_path))
                    self.summary_writer.update(np.mean(losses), np.sum(found_targets), i_episode, verbose=False)

                    if (i_episode + 1) % self.config.save_model_every == 0:
                        self.agent.store_model()

                    e_end_time = time.time()
                    print("episode {} spent {} secs".format(i_episode, e_end_time - e_start_time))

            if (i_episode + 1) % 5 == 0:
                self.agent.scheduler.step()
                print("============================learning rate:",
                      self.agent.q_network_optimizer.state_dict()['param_groups'][0]['lr'])
        result = [self.rewards_sum_n_episodes, self.found_targets_sum_n_episodes]
        result_sv_path = os.path.join(self.config.folder['out_folder'], "reward_found_targets.obj")
        pickle.dump(result, open(result_sv_path, "wb"))

    def print_info(self, i_episode, robot_pose, actions, rewards, found_targets, unknown_cells, known_cells, losses):
        print("\nepisode {} over".format(i_episode))
        print("End robot pose: {}".format(robot_pose[:3]))
        print("actions:{}".format(np.array(actions)))
        print("rewards:{}".format(np.array(rewards)))
        print("found_targets:{}".format(np.array(found_targets)))

        print(
            "Episode : {} | Mean loss : {} | Reward : {} | Found_targets : {} | unknown_cells :{}-{} | known cells :{}".format(
                i_episode,
                np.mean(losses),
                np.sum(rewards),
                np.sum(
                    found_targets), np.sum(unknown_cells), 0.008 * np.sum(unknown_cells),
                np.sum(known_cells)))
