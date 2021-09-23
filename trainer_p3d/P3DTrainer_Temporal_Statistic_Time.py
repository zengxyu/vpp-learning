import math
import os.path
import pickle

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
        self.rewards = []
        self.rewards_sum_n_episodes = []
        self.found_targets_sum_n_episodes = []
        self.taken_steps = []

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

                # 前5步，随便选择一个动作
                action = self.agent.pick_action([observed_map, self.deque.get_robot_poses()])
                (_, observed_map_next,
                 robot_pose_next), found_target_num, unknown_cells_num, known_cells_num, _, _, done, _ = self.field.step(
                    action)
                reward = found_target_num

                # 奖励unknown
                if is_reward_plus_unknown_cells:
                    reward += 0.006 * unknown_cells_num

                # 奖励前150step获得的targets
                if found_target_num == 0:
                    acc_convergence_reward = 0
                else:
                    acc_convergence_reward = (found_target_num / 1000 + 1) ** (5 - log2(step + 1))

                # 奖励好奇心，如果map差别比较大，那么奖励大
                map_diff_reward = 0
                if is_map_diff_reward:
                    map_diff_reward = np.sqrt(np.sum(np.square(observed_map_next - observed_map))) / 100
                    # print(map_diff_reward)

                if found_target_num == 0:
                    zero_found_target_consistent_count += 1
                else:
                    zero_found_target_consistent_count = 0

                negative_reward = 0
                if is_add_negative_reward:
                    negative_reward = -10 * zero_found_target_consistent_count
                # print(negative_reward)
                reward = reward + acc_convergence_reward + map_diff_reward + negative_reward
                reward = int(reward)

                # 如果连续N个奖励都是0，那么终止该序列，为了让它尽快找到
                if is_stop_n_zero_rewards:
                    # # reward redefine
                    if zero_found_target_consistent_count >= 30:
                        done = True

                # if robot_pose is the same with the robot_pose_next, then reward--
                robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ self.initial_direction

                # diff direction
                robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)

                self.deque.append_next(robot_pose_input_next)
                self.agent.step(state=[observed_map, self.deque.get_robot_poses()], action=action, reward=reward,
                                next_state=[observed_map_next, self.deque.get_next_robot_poses()], done=done)
                path.append(robot_pose)
                # to the next state
                observed_map = observed_map_next.copy()
                robot_pose = robot_pose_next.copy()
                # trainer_p3d
                if self.time_step % self.config.learn_every == 0 and self.config.is_train:
                    loss = self.agent.learn()

                actions.append(action)
                rewards.append(reward)
                found_targets.append(found_target_num)
                unknown_cells.append(unknown_cells_num)
                known_cells.append(known_cells_num)
                losses.append(loss)
                self.time_step += 1
                step += 1
                done = np.sum(found_targets) > 0.5 * 75370

                # record
                if not headless:
                    threading.Thread.considerYield()

                if done:
                    print("Taken {} steps".format(step))
                    self.taken_steps.append(step)
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

            # if (i_episode + 1) % 10 == 0:
            #     self.predict(i_episode, is_randomize, is_reward_plus_unknown_cells)
            if (i_episode + 1) % 5 == 0:
                self.agent.scheduler.step()
                print("============================learning rate:",
                      self.agent.q_network_optimizer.state_dict()['param_groups'][0]['lr'])
        result = self.taken_steps
        result_sv_path = os.path.join(self.config.folder['out_folder'], "steps.obj")
        pickle.dump(result, open(result_sv_path, "wb"))

    def predict(self, i_episode, is_randomize, is_reward_plus_unknown_cells):
        print("\ninference episode {}".format(i_episode))
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

        self.agent.reset()

        _, observed_map, robot_pose, _ = self.field.reset(is_randomize=is_randomize,
                                                          randomize_control=False,
                                                          randomize_from_48_envs=False,
                                                          is_save_env=False,
                                                          last_targets_found=self.last_targets_found)
        print("robot pose:{}".format(robot_pose))
        print("observation size:{}; robot pose size:{}".format(observed_map.shape, robot_pose.shape))
        relative_pose = np.zeros(shape=(6,))
        while not done:
            loss = 0

            # robot direction
            robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ self.initial_direction
            robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)

            self.deque.append(relative_pose)

            # 前5步，随便选择一个动作
            action = self.agent.pick_action([observed_map, self.deque.get_robot_poses()])
            (_, observed_map_next,
             robot_pose_next), found_target_num, unknown_cells_num, known_cells_num, _, _, done, _ = self.field.step(
                action)
            # 奖励
            reward = found_target_num
            # 奖励unknown
            if is_reward_plus_unknown_cells:
                reward += 0.006 * unknown_cells_num

            # if robot_pose is the same with the robot_pose_next, then reward--
            robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ self.initial_direction

            # diff direction
            robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)
            relative_pose_next = robot_pose_input_next - robot_pose_input

            self.deque.append_next(relative_pose_next)
            path.append(robot_pose)
            # to the next state
            observed_map = observed_map_next.copy()
            robot_pose = robot_pose_next.copy()
            relative_pose = relative_pose_next.copy()

            actions.append(action)
            rewards.append(reward)
            found_targets.append(found_target_num)
            unknown_cells.append(unknown_cells_num)
            known_cells.append(known_cells_num)
            losses.append(loss)
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
                self.summary_writer.update_inference_data(np.mean(losses), np.sum(found_targets), i_episode,
                                                          verbose=False)

                e_end_time = time.time()
                print("episode {} spent {} secs".format(i_episode, e_end_time - e_start_time))

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