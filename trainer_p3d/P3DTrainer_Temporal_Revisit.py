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

    def train(self, is_sph_pos, is_global_known_map, is_egocetric, is_randomize,
              is_reward_plus_unknown_cells, randomize_control, is_spacial, seq_len, is__save_path,
              is_stop_n_zero_rewards, is_map_diff_reward, is_add_negative_reward, is_revisit_penalty):
        self.seq_len = seq_len
        self.deque = Pose_State_DEQUE(capacity=self.seq_len)

        if headless:
            self.main_loop(is_sph_pos, is_global_known_map, is_egocetric, is_randomize, is_reward_plus_unknown_cells,
                           randomize_control, is_spacial, is__save_path, is_stop_n_zero_rewards, is_map_diff_reward,
                           is_add_negative_reward, is_revisit_penalty)
        else:
            # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
            # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
            main_thread = threading.Thread(target=self.main_loop)
            main_thread.start()
            self.field.gui.run()

    def main_loop(self, is_sph_pos, is_global_known_map, is_egocetric, is_randomize, is_reward_plus_unknown_cells,
                  randomize_control, is_spacial, is_save_path, is_stop_n_zero_rewards, is_map_diff_reward,
                  is_add_negative_reward, is_revisit_penalty):
        time_step = 0
        initial_direction = np.array([[1], [0], [0]])
        last_targets_found = 0
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
            revisit_penalties = []
            actions = []
            path = []
            step = 0
            zero_found_target_consistent_count = 0

            self.agent.reset()

            revisit_map, observed_map, robot_pose = self.field.reset(is_sph_pos=is_sph_pos,
                                                                     is_global_known_map=is_global_known_map,
                                                                     is_egocetric=is_egocetric,
                                                                     is_randomize=is_randomize,
                                                                     randomize_control=randomize_control,
                                                                     is_spacial=is_spacial,
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
                action = self.agent.pick_action([revisit_map, observed_map, self.deque.get_robot_poses()])
                (revisit_map_next, observed_map_next,
                 robot_pose_next), found_target_num, unknown_cells_num, known_cells_num, revisit_penalty, done, _ = self.field.step(
                    action)

                # a = found_target_num
                # b = 0.05 * unknown_cells_num ** (
                #         1 - step / self.max_steps / 2)
                # c = - (known_cells_num / 2000) ** 1.1
                reward = found_target_num

                # 奖励unknown
                if is_reward_plus_unknown_cells:
                    reward += 0.008 * unknown_cells_num

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
                revisit_penalty2 = 0
                if is_revisit_penalty:
                    revisit_penalty2 = 10 * revisit_penalty
                    # print(revisit_penalty)
                reward = reward + acc_convergence_reward + map_diff_reward + negative_reward + revisit_penalty2
                reward = int(reward)

                # 如果连续N个奖励都是0，那么终止该序列，为了让它尽快找到
                if is_stop_n_zero_rewards:
                    # # reward redefine
                    if zero_found_target_consistent_count >= 30:
                        done = True

                # if robot_pose is the same with the robot_pose_next, then reward--
                robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ initial_direction

                # diff direction
                robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)

                self.deque.append_next(robot_pose_input_next)
                self.agent.step(state=[revisit_map.astype(np.int8), observed_map, self.deque.get_robot_poses()],
                                action=action,
                                reward=reward,
                                next_state=[revisit_map_next.astype(np.int8), observed_map_next,
                                            self.deque.get_next_robot_poses()],
                                done=done)
                path.append(robot_pose)
                # to the next state
                observed_map = observed_map_next.copy()
                robot_pose = robot_pose_next.copy()
                revisit_map = revisit_map_next.copy()
                # trainer_p3d
                if time_step % self.config.learn_every == 0 and self.config.is_train:
                    for i in range(20):
                        loss = self.agent.learn()

                actions.append(action)
                rewards.append(reward)
                found_targets.append(found_target_num)
                unknown_cells.append(unknown_cells_num)
                known_cells.append(known_cells_num)
                revisit_penalties.append(revisit_penalty)
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
                    paths.append(path.copy())

                    print("\nepisode {} over".format(i_episode))
                    print("robot pose: {}".format(robot_pose[:3]))
                    print("actions:{}".format(np.array(actions)))
                    print("rewards:{}".format(np.array(rewards)))
                    print("found_targets:{}".format(np.array(found_targets)))

                    print(
                        "Episode : {} | Mean loss : {} | Reward : {} | Found_targets : {} | unknown_cells :{}-{} | known cells :{} | penalty:{}".format(
                            i_episode,
                            np.mean(losses),
                            np.sum(rewards),
                            np.sum(
                                found_targets), np.sum(unknown_cells), 0.008 * np.sum(unknown_cells),
                            np.sum(known_cells), np.sum(revisit_penalties)))
                    if is_save_path:
                        file_path = os.path.join(self.config.folder["out_folder"], "path.obj")
                        pickle.dump(paths, open(file_path, "wb"))
                        print("save robot path to file_path:{}".format(file_path))
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
