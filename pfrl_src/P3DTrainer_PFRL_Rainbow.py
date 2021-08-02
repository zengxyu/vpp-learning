import logging
from typing import List, Dict, Any, Optional

import numpy as np
import time

import pfrl
import torch
from pfrl import explorers, replay_buffers, agents
from pfrl.replay_buffer import batch_experiences
from pfrl.replay_buffers import PrioritizedReplayBuffer
from scipy.spatial.transform.rotation import Rotation

import action_space
from utilities.basic_logger import BasicLogger
from utilities.summary_writer import SummaryWriterLogger
from network.network_dqn import DQN_Network11_PFRL_Rainbow

headless = True
if not headless:
    from direct.stdpy import threading


class MyDQN(agents.CategoricalDoubleDQN):
    def update(
            self, experiences: List[List[Dict[str, Any]]], errors_out: Optional[list] = None
    ) -> None:
        """Update the model from experiences

        Args:
            experiences (list): List of lists of dicts.
                For DQN, each dict must contains:
                  - state (object): State
                  - action (object): Action
                  - reward (float): Reward
                  - is_state_terminal (bool): True iff next state is terminal
                  - next_state (object): Next state
                  - weight (float, optional): Weight coefficient. It can be
                    used for importance sampling.
            errors_out (list or None): If set to a list, then TD-errors
                computed from the given experiences are appended to the list.

        Returns:
            None
        """
        has_weight = "weight" in experiences[0][0]
        exp_batch = batch_experiences(
            experiences,
            device=self.device,
            phi=self.phi,
            gamma=self.gamma,
            batch_states=self.batch_states,
        )
        if has_weight:
            exp_batch["weights"] = torch.tensor(
                [elem[0]["weight"] for elem in experiences],
                device=self.device,
                dtype=torch.float32,
            )
            if errors_out is None:
                errors_out = []
        loss = self._compute_loss(exp_batch, errors_out=errors_out)
        rewards = exp_batch['reward'].numpy()
        errors_out = abs(rewards) + abs(np.array(errors_out))
        if has_weight:
            assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
            # TODO 修改error
            # reward = experiences
            # errors_out = abs(errors_out)+abs()
            self.replay_buffer.update_errors(errors_out)

        self.loss_record.append(float(loss.detach().cpu().numpy()))

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            pfrl.utils.clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optim_t += 1


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
        q_func = DQN_Network11_PFRL_Rainbow(0, n_actions)

        # Noisy nets
        pfrl.nn.to_factorized_noisy(q_func, sigma_scale=0.1)
        # Turn off explorer
        explorer = explorers.ExponentialDecayEpsilonGreedy(start_epsilon=0.5, end_epsilon=0.1, decay=0.99975,
                                                           random_action_func=self.field.action_space.sample)

        # Use the same eps as https://arxiv.org/abs/1710.02298
        opt = torch.optim.Adam(q_func.parameters(), 1e-4, eps=1.5e-4)

        # Prioritized Replay
        # Anneal beta from beta0 to 1 throughout training
        update_interval = 1
        n_step_return = 3
        gpu = -1
        gamma = 0.9
        replay_start_size = 50
        betasteps = 2 * 10 ** 6 / update_interval
        rbuf = replay_buffers.PrioritizedReplayBuffer(
            10 ** 6,
            alpha=0.5,
            beta0=0.4,
            betasteps=betasteps,
            num_steps=n_step_return,
            normalize_by_max=True,
        )

        def phi(x):
            frame, robot_pose = x
            frame = frame.astype(np.float32, copy=False)
            robot_pose = robot_pose.astype(np.float32, copy=False)
            return (frame, robot_pose)

        agent = MyDQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=gamma,
            explorer=explorer,
            minibatch_size=32,
            replay_start_size=replay_start_size,
            target_update_interval=2000,
            update_interval=update_interval,
            batch_accumulator="mean",
            phi=phi,
            max_grad_norm=10,
        )

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
