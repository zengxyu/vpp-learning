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

        self.global_time_step = 0
        self.initial_direction = np.array([[1], [0], [0]])
        self.last_targets_found = 0

    def train(self, is_randomize, is_reward_plus_unknown_cells, randomize_control, seq_len, is_save_path,
              is_stop_n_zero_rewards, is_map_diff_reward, is_add_negative_reward):
        self.seq_len = seq_len
        self.deque = Pose_State_DEQUE(capacity=self.seq_len)

        if headless:
            self.main_loop(is_randomize, is_reward_plus_unknown_cells, randomize_control, is_save_path)
        else:
            # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
            # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
            main_thread = threading.Thread(target=self.main_loop)
            main_thread.start()
            self.field.gui.run()

    def main_loop(self, is_randomize, is_reward_plus_unknown_cells, randomize_control, is_save_path):

        paths = []
        for i_episode in range(self.config.num_episodes_to_run):
            print("\ntraining episode {}".format(i_episode))
            e_start_time = time.time()
            done = False
            losses = []
            losses_unknowns = []
            losses_knowns = []
            exploit_steps = []
            explore_steps = []
            found_targets = []
            unknown_cells = []
            known_cells = []
            actions = []
            path = []
            step = 0
            loss_unknown = 0
            loss_known = 0
            self.agent.reset()
            relative_pose = np.zeros(shape=(6,))
            _, observed_map, robot_pose, (known_target_rate, unknown_rate) = self.field.reset(is_randomize=is_randomize,
                                                                                              randomize_control=randomize_control,
                                                                                              last_targets_found=self.last_targets_found)
            print("observation size:{}; robot pose size:{}".format(observed_map.shape, robot_pose.shape))
            exploit = False
            while not done:
                # robot direction
                robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ self.initial_direction
                robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)

                self.deque.append(relative_pose)
                # 如果是在探索，那么known_target_rate > 0.1 and unknown_rate > 0.2利用
                if not exploit:
                    exploit = known_target_rate > 0.01 and unknown_rate > 0.2
                #  如果是在利用，那么unknown_rate < 0.2 转为探索
                if exploit and unknown_rate < 0.3:
                    exploit = False

                if exploit:
                    action = self.agent.pick_action_known([observed_map, self.deque.get_robot_poses()])
                    exploit_steps.append(step)
                    explore_steps.append(0)
                else:
                    action = self.agent.pick_action_unknown([observed_map, self.deque.get_robot_poses()])
                    explore_steps.append(step)
                    exploit_steps.append(0)
                (_, observed_map_next,
                 robot_pose_next), found_target_num, unknown_cells_num, known_cells_num, _, (
                    known_target_rate, unknown_rate), done, _ = self.field.step(action)

                reward = [unknown_cells_num, found_target_num]

                # if robot_pose is the same with the robot_pose_next, then reward--
                robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ self.initial_direction

                # diff direction
                robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)
                relative_pose_next = robot_pose_input_next - robot_pose_input

                self.deque.append_next(relative_pose_next)
                self.agent.step(state=[observed_map, self.deque.get_robot_poses()], action=action, reward=reward,
                                next_state=[observed_map_next, self.deque.get_next_robot_poses()], done=done)
                # path.append(robot_pose)
                # to the next state
                observed_map = observed_map_next.copy()
                robot_pose = robot_pose_next.copy()
                relative_pose = relative_pose_next.copy()
                # trainer_p3d
                if self.global_time_step % 20 == 0 and self.config.is_train:
                    for i in range(12):
                        loss_unknown, loss_known = self.agent.learn()

                actions.append(action)
                # rewards.append(reward)
                losses_unknowns.append(loss_unknown)
                losses_knowns.append(loss_known)
                found_targets.append(found_target_num)
                unknown_cells.append(unknown_cells_num)
                known_cells.append(known_cells_num)
                self.global_time_step += 1
                step += 1
                # record
                if not headless:
                    threading.Thread.considerYield()

                if done:
                    step = 0
                    self.deque.clear()
                    self.last_targets_found = np.sum(found_targets)
                    # paths.append(path.copy())
                    print("it explore :{} ".format(explore_steps))
                    print("it exploit :{} ".format(exploit_steps))
                    print("it explore :{} times".format(np.count_nonzero(explore_steps)))
                    print("it exploit :{} times".format(np.count_nonzero(exploit_steps)))
                    self.print_info(i_episode, robot_pose, actions, found_targets, found_targets, unknown_cells,
                                    known_cells,
                                    losses_unknowns, losses_knowns)

                    if is_save_path:
                        file_path = os.path.join(self.config.folder["out_folder"], "path.obj")
                        pickle.dump(paths, open(file_path, "wb"))
                        print("save robot path to file_path:{}".format(file_path))
                    self.summary_writer.update(np.mean(losses), np.sum(found_targets), i_episode, verbose=False)

                    if (i_episode + 1) % self.config.save_model_every == 0:
                        self.agent.store_model()

                    e_end_time = time.time()
                    print("episode {} spent {} secs".format(i_episode, e_end_time - e_start_time))
            if (i_episode + 1) % 1 == 0:
                self.predict(i_episode, is_randomize, is_reward_plus_unknown_cells)

    def predict(self, i_episode, is_randomize, is_reward_plus_unknown_cells):
        print("\ntraining episode {}".format(i_episode))
        e_start_time = time.time()
        done = False
        losses = []
        losses_unknowns = []
        losses_knowns = []
        exploit_steps = []
        explore_steps = []
        found_targets = []
        unknown_cells = []
        known_cells = []
        actions = []
        path = []
        step = 0
        loss_unknown = 0
        loss_known = 0
        self.agent.reset()
        relative_pose = np.zeros(shape=(6,))
        _, observed_map, robot_pose, (known_target_rate, unknown_rate) = self.field.reset(is_randomize=is_randomize,
                                                                                          randomize_control=False,
                                                                                          last_targets_found=self.last_targets_found)
        print("observation size:{}; robot pose size:{}".format(observed_map.shape, robot_pose.shape))
        exploit = False
        while not done:
            # robot direction
            robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ self.initial_direction
            robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)

            self.deque.append(relative_pose)
            # 如果是在探索，那么known_target_rate > 0.1 and unknown_rate > 0.2利用
            if not exploit:
                exploit = known_target_rate > 0.01 and unknown_rate > 0.2
            #  如果是在利用，那么unknown_rate < 0.2 转为探索
            if exploit and unknown_rate < 0.3:
                exploit = False

            if exploit:
                action = self.agent.pick_action_known([observed_map, self.deque.get_robot_poses()])
                exploit_steps.append(step)
                explore_steps.append(0)
            else:
                action = self.agent.pick_action_unknown([observed_map, self.deque.get_robot_poses()])
                explore_steps.append(step)
                exploit_steps.append(0)
            (_, observed_map_next,
             robot_pose_next), found_target_num, unknown_cells_num, known_cells_num, _, (
                known_target_rate, unknown_rate), done, _ = self.field.step(action)

            # if robot_pose is the same with the robot_pose_next, then reward--
            robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ self.initial_direction

            # diff direction
            robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)
            relative_pose_next = robot_pose_input_next - robot_pose_input

            self.deque.append_next(relative_pose_next)
            observed_map = observed_map_next.copy()
            robot_pose = robot_pose_next.copy()
            relative_pose = relative_pose_next.copy()

            actions.append(action)
            # rewards.append(reward)
            losses_unknowns.append(loss_unknown)
            losses_knowns.append(loss_known)
            found_targets.append(found_target_num)
            unknown_cells.append(unknown_cells_num)
            known_cells.append(known_cells_num)
            self.global_time_step += 1
            step += 1
            # record
            if not headless:
                threading.Thread.considerYield()

            if done:
                step = 0
                self.deque.clear()
                self.last_targets_found = np.sum(found_targets)
                # paths.append(path.copy())
                print("it explore :{} ".format(explore_steps))
                print("it exploit :{} ".format(exploit_steps))
                print("it explore :{} times".format(np.count_nonzero(explore_steps)))
                print("it exploit :{} times".format(np.count_nonzero(exploit_steps)))
                self.print_info(i_episode, robot_pose, actions, found_targets, found_targets, unknown_cells,
                                known_cells,
                                losses_unknowns, losses_knowns)

                self.summary_writer.update_inference_data(np.mean(losses), np.sum(found_targets), i_episode, verbose=False)

                if (i_episode + 1) % self.config.save_model_every == 0:
                    self.agent.store_model()

                e_end_time = time.time()
                print("episode {} spent {} secs".format(i_episode, e_end_time - e_start_time))

    def print_info(self, i_episode, robot_pose, actions, rewards, found_targets, unknown_cells, known_cells,
                   losses_unknowns, losses_knowns):
        print("\nepisode {} over".format(i_episode))
        print("End robot pose: {}".format(robot_pose[:3]))
        print("actions:{}".format(np.array(actions)))
        print("rewards:{}".format(np.array(rewards)))
        print("found_targets:{}".format(np.array(found_targets)))

        print(
            "Episode : {} | Mean losses_unknowns : {}  | Mean losses_knowns : {}  | Reward : {} | Found_targets : {} | unknown_cells :{}-{} | known cells :{}".format(
                i_episode,
                np.mean(losses_unknowns),
                np.mean(losses_knowns),
                np.sum(rewards),
                np.sum(
                    found_targets), np.sum(unknown_cells), 0.008 * np.sum(unknown_cells),
                np.sum(known_cells)))
