import logging
import os
import random
from collections import deque

import numpy as np
import time
from scipy.spatial.transform.rotation import Rotation

from utilities.basic_logger import BasicLogger
from utilities.summary_writer import SummaryWriterLogger
from ray import tune

headless = True
if not headless:
    from direct.stdpy import threading


class State_DEQUE:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states_deque = deque([])
        self.next_states_deque = deque([])
        self.robot_poses = deque([])
        self.next_robot_poses = deque([])

    def __len__(self):
        assert self.states_deque.__len__() == self.next_states_deque.__len__()
        assert self.robot_poses.__len__() == self.next_robot_poses.__len__()
        assert self.states_deque.__len__() == self.robot_poses.__len__()
        return self.states_deque.__len__()

    def append(self, state, robot_pose):
        self.states_deque.append(state)
        self.robot_poses.append(robot_pose)
        if self.states_deque.__len__() > self.capacity:
            self.states_deque.popleft()
            self.robot_poses.popleft()

    def append_next(self, next_state, next_robot_pose):
        self.next_states_deque.append(next_state)
        self.next_robot_poses.append(next_robot_pose)
        if self.next_states_deque.__len__() > self.capacity:
            self.next_states_deque.popleft()
            self.next_robot_poses.popleft()

    def get_states(self):
        states = np.array(self.states_deque, dtype=float)
        return states

    def get_next_states(self):
        next_states = np.array(self.next_states_deque, dtype=float)
        return next_states

    def get_robot_poses(self):
        robot_poses = np.array(self.robot_poses, dtype=float)
        return robot_poses

    def get_next_robot_poses(self):
        next_robot_poses = np.array(self.next_robot_poses, dtype=float)
        return next_robot_poses

    def is_full(self):
        return self.__len__() == self.capacity


class P3DTrainer(object):
    def __init__(self, config, Agent, Field, Action, project_path):
        self.config = config
        self.Agent = Agent
        self.Field = Field
        self.Action = Action

        self.summary_writer = SummaryWriterLogger(config)
        init_file_path = os.path.join(project_path, 'VG07_6.binvox')
        # field
        self.field = self.Field(Action=Action, shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05,
                                max_steps=300, init_file=init_file_path, headless=headless)

        self.config.environment = {
            "is_vpp": True,
            "reward_threshold": 0,
            "state_size": self.get_state_size(self.field),
            "action_size": self.get_action_size(self.field),
            "action_shape": self.get_action_size(self.field),
        }
        # Agent
        self.agent = self.Agent(self.config)
        self.seq_len = 5
        self.logger = BasicLogger.setup_console_logging(config)
        self.deque = State_DEQUE(capacity=self.seq_len)

    def train(self):
        if headless:
            self.main_loop()
        else:
            # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
            # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
            main_thread = threading.Thread(target=self.main_loop)
            main_thread.start()
            self.field.gui.run()

    def imitation_learning(self):
        pass

    def main_loop(self):
        observed_map, robot_pose = self.field.reset()
        print("observation size:{}; robot pose size:{}".format(observed_map.shape, robot_pose.shape))
        time_step = 0
        initial_direction = np.array([[1], [0], [0]])
        mean_loss_last_n_ep, mean_reward_last_n_ep = 0, 0
        # 只能包含
        for i_episode in range(self.config.num_episodes_to_run):
            print("\nepisode {}".format(i_episode))
            e_start_time = time.time()
            done = False
            losses = []
            rewards = []
            found_target_nums = []
            unknown_cell_nums = []
            actions = []
            self.agent.reset()
            observed_map, robot_pose = self.field.reset()
            relative_robot_pose = np.zeros(shape=(6,))
            while not done:
                loss = 0

                # robot direction
                robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ initial_direction
                robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)
                self.deque.append(observed_map, relative_robot_pose)

                # 前5步，随便选择一个动作
                if time_step <= self.seq_len:
                    action = random.randint(0, 11)
                else:
                    action = self.agent.pick_action([self.deque.get_states(), self.deque.get_robot_poses()])

                (observed_map_next, robot_pose_next), found_target_num, unknown_cell_num, done, _ = self.field.step(
                    action)
                reward = found_target_num

                # if robot_pose is the same with the robot_pose_next, then reward--
                robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ initial_direction
                # diff direction
                robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)

                relative_pose_input_next = robot_pose_input_next - robot_pose_input
                self.deque.append_next(observed_map_next, relative_pose_input_next)
                self.agent.step(state=[observed_map, relative_robot_pose], action=action, reward=reward,
                                next_state=[observed_map_next, relative_pose_input_next], done=done)

                # if self.deque.is_full():
                #     self.agent.step(state=[self.deque.get_states(), self.deque.get_robot_poses()], action=action,
                #                     reward=reward,
                #                     next_state=[self.deque.get_next_states(), self.deque.get_next_robot_poses()],
                #                     done=done)

                # to the next state
                observed_map = observed_map_next.copy()
                robot_pose = robot_pose_next.copy()
                relative_robot_pose = relative_pose_input_next.copy()
                # train
                if time_step % self.config.learn_every == 0 and self.deque.is_full():
                    loss = self.agent.learn()

                actions.append(action)
                found_target_nums.append(found_target_num)
                unknown_cell_nums.append(unknown_cell_num)
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
                    print("found_target_nums:{}".format(np.array(found_target_nums)))
                    # print("unknown_cell_nums:{}".format(np.array(unknown_cell_nums)))

                    mean_loss_last_n_ep, mean_reward_last_n_ep = self.summary_writer.update(np.mean(losses),
                                                                                            np.sum(found_target_nums),
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
