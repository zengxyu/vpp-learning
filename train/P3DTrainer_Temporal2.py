import logging
import os
import random

import numpy as np
import time
from scipy.spatial.transform.rotation import Rotation

from train.StateDeque import State_DEQUE
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
        self.deque = State_DEQUE(capacity=self.seq_len)

    def train(self, is_sph_pos=False, is_global_known_map=False, is_randomize=False,
              is_reward_plus_unknown_cells=False):
        if headless:
            self.main_loop(is_sph_pos, is_global_known_map, is_randomize, is_reward_plus_unknown_cells)
        else:
            # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
            # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
            main_thread = threading.Thread(target=self.main_loop)
            main_thread.start()
            self.field.gui.run()

    def main_loop(self, is_sph_pos, is_global_known_map, is_randomize, is_reward_plus_unknown_cells):
        time_step = 0
        initial_direction = np.array([[1], [0], [0]])
        mean_loss_last_n_ep, mean_reward_last_n_ep = 0, 0
        for i_episode in range(self.config.num_episodes_to_run):
            print("\nepisode {}".format(i_episode))
            e_start_time = time.time()
            done = False
            losses = []
            rewards = []
            actions = []
            self.agent.reset()
            known_map, observed_map, robot_pose = self.field.reset(is_sph_pos=is_sph_pos,
                                                                   is_global_known_map=is_global_known_map,
                                                                   is_randomize=is_randomize)
            known_map_next = None
            print("robot pose:{}".format(robot_pose))
            print("observation size:{}; robot pose size:{}".format(observed_map.shape, robot_pose.shape))
            while not done:
                loss = 0

                # robot direction
                robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ initial_direction
                robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)
                self.deque.append(observed_map, robot_pose_input)

                # 前5步，随便选择一个动作
                if time_step <= self.seq_len:
                    action = random.randint(0, 11)
                else:
                    action = self.agent.pick_action([known_map, self.deque.get_states(), self.deque.get_robot_poses()])
                if is_global_known_map:
                    (known_map_next, observed_map_next,
                     robot_pose_next), found_targets, unknown_cells, done, _ = self.field.step(action)
                else:
                    (observed_map_next, robot_pose_next), found_targets, unknown_cells, done, _ = self.field.step(
                        action)
                if is_reward_plus_unknown_cells:
                    reward = found_targets + 0.001 * unknown_cells
                else:
                    reward = found_targets
                # if robot_pose is the same with the robot_pose_next, then reward--
                robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ initial_direction

                # diff direction
                robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)

                self.deque.append_next(observed_map_next, robot_pose_input_next)

                if is_global_known_map:
                    self.agent.step(state=[known_map, observed_map, robot_pose_input], action=action,
                                    reward=reward,
                                    next_state=[known_map_next, observed_map_next, robot_pose_input_next], done=done)
                else:
                    self.agent.step(state=[observed_map, robot_pose_input], action=action, reward=reward,
                                    next_state=[observed_map_next, robot_pose_input_next], done=done)

                # to the next state
                observed_map = observed_map_next.copy()
                robot_pose = robot_pose_next.copy()
                known_map = known_map_next.copy()
                # train
                if time_step % self.config.learn_every == 0 and time_step > self.seq_len:
                    loss = self.agent.learn()

                actions.append(action)
                rewards.append(reward)
                losses.append(loss)
                time_step += 1
                # print(
                #     "{}-th episode : {}-th step takes {} secs; action:{}; found target:{}; sum found targets:{}; reward:{}; sum reward:{}".format(
                #         i_episode,
                #         step_count,
                #         time.time() - time3,
                #         action, found_target, np.sum(found_targets) + found_target, reward,
                #         np.sum(rewards) + reward))
                # record
                if not headless:
                    threading.Thread.considerYield()

                if done:
                    print("\nepisode {} over".format(i_episode))
                    print("robot pose: {}".format(robot_pose[:3]))
                    print("actions:{}".format(np.array(actions)))
                    print("rewards:{}".format(np.array(rewards)))
                    mean_loss_last_n_ep, mean_reward_last_n_ep = self.summary_writer.update(np.mean(losses),
                                                                                            np.sum(rewards),
                                                                                            i_episode)

                    if (i_episode + 1) % self.config.save_model_every == 0:
                        self.agent.store_model()

                    e_end_time = time.time()
                    print("episode {} spent {} secs".format(i_episode, e_end_time - e_start_time))
        print('Complete')

    def get_state_size(self, field):
        return 0

    def get_action_size(self, field):
        return field.get_action_size()
