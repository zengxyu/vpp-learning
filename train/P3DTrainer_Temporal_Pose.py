import logging
import os
import random

import numpy as np
import time
from scipy.spatial.transform.rotation import Rotation

from train.StateDeque import Pose_State_DEQUE
from utilities.basic_logger import BasicLogger
from utilities.summary_writer import SummaryWriterLogger

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
        self.deque = Pose_State_DEQUE(capacity=self.seq_len)
        self.max_steps = self.field.max_steps

    def train(self, is_sph_pos, is_global_known_map, is_egocetric, is_randomize,
              is_reward_plus_unknown_cells, randomize_control):
        if headless:
            self.main_loop(is_sph_pos, is_global_known_map, is_egocetric, is_randomize, is_reward_plus_unknown_cells,
                           randomize_control)
        else:
            # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
            # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
            main_thread = threading.Thread(target=self.main_loop)
            main_thread.start()
            self.field.gui.run()

    def main_loop(self, is_sph_pos, is_global_known_map, is_egocetric, is_randomize, is_reward_plus_unknown_cells,
                  randomize_control):
        time_step = 0
        initial_direction = np.array([[1], [0], [0]])
        last_targets_found = 0
        for i_episode in range(self.config.num_episodes_to_run):
            print("\nepisode {}".format(i_episode))
            e_start_time = time.time()
            done = False
            losses = []
            rewards = []
            found_targets = []
            unknown_cells = []
            actions = []
            step = 0
            self.agent.reset()

            observed_map, robot_pose = self.field.reset(is_sph_pos=is_sph_pos,
                                                        is_global_known_map=is_global_known_map,
                                                        is_egocetric=is_egocetric,
                                                        is_randomize=is_randomize,
                                                        randomize_control=randomize_control,
                                                        last_targets_found=last_targets_found)
            print("robot pose:{}".format(robot_pose))
            print("observation size:{}; robot pose size:{}".format(observed_map.shape, robot_pose.shape))
            while not done:
                loss = 0

                # robot direction
                robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ initial_direction
                robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)
                self.deque.append(robot_pose_input)

                # 前5步，随便选择一个动作
                if step <= self.seq_len:
                    action = random.randint(0, self.field.get_action_size())
                else:
                    action = self.agent.pick_action([observed_map, self.deque.get_robot_poses()])
                (observed_map_next,
                 robot_pose_next), found_target_num, unknown_cells_num, known_cells_num, done, _ = self.field.step(
                    action)
                if is_reward_plus_unknown_cells:
                    reward = found_target_num + 0.005 * unknown_cells_num ** (
                                1 - step / self.max_steps) - known_cells_num ** 1.1
                else:
                    reward = found_target_num
                # if robot_pose is the same with the robot_pose_next, then reward--
                robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ initial_direction

                # diff direction
                robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)

                self.deque.append_next(robot_pose_input_next)

                self.agent.step(state=[observed_map, robot_pose_input], action=action, reward=reward,
                                next_state=[observed_map_next, robot_pose_input_next], done=done)

                # to the next state
                observed_map = observed_map_next.copy()
                robot_pose = robot_pose_next.copy()
                # train
                if time_step % self.config.learn_every == 0 and time_step > self.seq_len:
                    loss = self.agent.learn()

                actions.append(action)
                rewards.append(reward)
                found_targets.append(found_target_num)
                unknown_cells.append(0.001 * unknown_cells_num)
                losses.append(loss)
                time_step += 1
                step += 1
                # record
                if not headless:
                    threading.Thread.considerYield()

                if done:
                    step = 0
                    self.deque.clear()
                    last_targets_found = np.sum(found_targets)
                    print("\nepisode {} over".format(i_episode))
                    print("robot pose: {}".format(robot_pose[:3]))
                    print("actions:{}".format(np.array(actions)))
                    print("rewards:{}".format(np.array(rewards)))
                    print("found_targets:{}".format(np.array(found_targets)))
                    print("Episode : {} | Mean loss : {} | Reward : {} | Found_targets : {} | unknown_cells :{}".format(
                        i_episode,
                        np.mean(losses),
                        np.sum(rewards),
                        np.sum(
                            found_targets), np.sum(unknown_cells)))

                    mean_loss_last_n_ep, mean_reward_last_n_ep = self.summary_writer.update(np.mean(losses),
                                                                                            np.sum(found_targets),
                                                                                            i_episode, verbose=False)

                    if (i_episode + 1) % self.config.save_model_every == 0:
                        self.agent.store_model()

                    e_end_time = time.time()
                    print("episode {} spent {} secs".format(i_episode, e_end_time - e_start_time))
        print('Complete')

    def get_state_size(self, field):
        return 0

    def get_action_size(self, field):
        return field.get_action_size()
