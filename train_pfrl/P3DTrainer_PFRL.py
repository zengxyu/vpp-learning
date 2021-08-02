import logging

import numpy as np
import time

import pfrl
import torch
from scipy.spatial.transform.rotation import Rotation

import action_space
from utilities.basic_logger import BasicLogger
from utilities.summary_writer import SummaryWriterLogger
from network.network_dqn import DQN_Network11_PFRL

headless = True
if not headless:
    from direct.stdpy import threading


class P3DTrainer_PFRL(object):
    def __init__(self, config, Field, Action):
        self.config = config
        self.Field = Field
        self.Action = Action

        self.summary_writer = SummaryWriterLogger(config)
        # field
        self.field = self.Field(Action=Action, shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05,
                                max_steps=300, init_file='VG07_6.binvox', headless=headless)

        self.config.environment = {
            "is_vpp": True,
            "reward_threshold": 0,
            "state_size": self.get_state_size(self.field),
            "action_size": self.get_action_size(self.field),
            "action_shape": self.get_action_size(self.field),
        }
        # Agent
        # self.agent = self.Agent(self.config)

        self.logger = BasicLogger.setup_console_logging(config)

    def train(self):
        if headless:
            self.main_loop()
        else:
            # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
            # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
            main_thread = threading.Thread(target=self.main_loop)
            main_thread.start()
            self.field.gui.run()

    def main_loop(self):
        # obs_size = self.field.observation_space.low.size
        n_actions = self.field.action_space.n
        q_func = DQN_Network11_PFRL(0, n_actions)

        # Use Adam to optimize q_func. eps=1e-2 is for stability.
        optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)
        # Set the discount factor that discounts future rewards.
        gamma = 0.9

        # Use epsilon-greedy for exploration
        explorer = pfrl.explorers.ConstantEpsilonGreedy(
            epsilon=0.3, random_action_func=self.field.action_space.sample)

        # DQN uses Experience Replay.
        # Specify a replay buffer and its capacity.
        replay_buffer = pfrl.replay_buffers.PrioritizedReplayBuffer(capacity=10 ** 6)

        # Since observations from CartPole-v0 is numpy.float64 while
        # As PyTorch only accepts numpy.float32 by default, specify
        # a converter as a feature extractor function phi.
        phi = lambda x: x.astype(np.float32, copy=False)

        def phi2(x):
            frame, robot_pose = x
            frame = frame.astype(np.float32, copy=False)
            robot_pose = robot_pose.astype(np.float32, copy=False)
            return (frame, robot_pose)

        # Set the device id to use GPU. To use CPU only, set it to -1.
        gpu = -1

        # Now create an agent that will interact with the environment.
        agent = pfrl.agents.DoubleDQN(
            q_func,
            optimizer,
            replay_buffer,
            gamma,
            explorer,
            replay_start_size=500,
            update_interval=1,
            target_update_interval=100,
            phi=phi2,
            gpu=gpu,
        )
        n_episodes = 300
        max_episode_len = 200

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
            observed_map, robot_pose = self.field.reset()
            t = 0  # time step

            while not done:
                loss = 0
                t += 1

                # robot direction
                robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ initial_direction
                robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)

                action = agent.act((observed_map, robot_pose_input))

                (observed_map_next, robot_pose_next), reward, done, _ = self.field.step(action)

                # if robot_pose is the same with the robot_pose_next, then reward--
                robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ initial_direction

                # diff direction
                robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)
                reset = t == max_episode_len

                agent.observe([observed_map, robot_pose_input], reward, done, reset)

                # to the next state
                observed_map = observed_map_next.copy()
                robot_pose = robot_pose_next.copy()

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
                    print("statistic:{}".format(agent.get_statistics()))
                    # if (i_episode + 1) % self.config.save_model_every == 0:
                    #     self.agent.store_model()

                    e_end_time = time.time()
                    print("episode {} spent {} secs".format(i_episode, e_end_time - e_start_time))
        print('Complete')

    def get_state_size(self, field):
        return 0

    def get_action_size(self, field):
        return field.get_action_size()
