import logging
import random
import time
from typing import Dict, List

import numpy as np
from direct.stdpy import threading

from agents.policies import Policies
from trainer.trainer_helper import add_scalar
from utilities.info import EpisodeInfo, InfoCollector
from torch.utils.tensorboard import SummaryWriter


class P3DTrainer(object):
    def __init__(self, env, agent, scheduler, action_space, parser_args):
        self.parser_args = parser_args
        self.training_config = parser_args.training_config
        self.writer = SummaryWriter(log_dir=parser_args.out_board)
        self.env = env
        self.agent = agent
        self.scheduler = scheduler
        self.action_space = action_space

        self.train_i_episode = 0
        self.test_i_episode = 0
        self.global_i_step = 0
        self.train_collector = InfoCollector(self.training_config["train_smooth_n"])
        self.test_collector = InfoCollector(self.training_config["test_smooth_n"])
        self.eval = not parser_args.train

    def run(self):
        print("========================================Start running========================================")
        # discrete
        if self.parser_args.policy == Policies.RL_Policy and (self.eval or self.parser_args.resume):
            logging.info("load model from {} {}".format(self.parser_args.in_model, self.parser_args.in_model_index))
            self.agent.load("{}/model_epi_{}".format(self.parser_args.in_model, self.parser_args.in_model_index))

        head = self.parser_args.head
        if self.parser_args.train and not head:
            print("Start training")
            for i in range(self.training_config["num_episodes"]):
                print("\nEpisode:{}".format(i))
                self.train_once()
                if (i + 1) % 10 == 0:
                    print("\nTest Episode:{}".format(i))
                    self.evaluate_once()
                self.scheduler.step()
                print("Current learning rate : {}".format(self.agent.optimizer.state_dict()['param_groups'][0]['lr']))
        else:
            if head:
                print("Start evaluating with head")
                main_thread = threading.Thread(target=self.evaluate_n_times,
                                               args=(self.parser_args.epsilon_greedy, self.parser_args.epsilon,
                                                     self.parser_args.training_config["num_episodes"]))
                main_thread.start()
                self.env.gui.run()
            else:
                print("Start evaluating without head")
                self.evaluate_n_times(self.parser_args.epsilon_greedy, self.parser_args.epsilon,
                                      self.parser_args.training_config["num_episodes"])

    def train_once(self):
        phase = "Train"
        start_time = time.time()

        self.train_i_episode += 1
        state, _ = self.env.reset()

        done = False
        infos = []
        i_step = 0
        while not done:
            action = self.agent.act(state)
            state, reward, done, info = self.env.step(action)
            self.agent.observe(obs=state, reward=reward, done=done, reset=False)
            self.global_i_step += 1
            i_step += 1
            infos.append(info)

        self.train_collector.add(infos)
        smooth_results = self.train_collector.get_p3d_smooth_statistic(self.env, self.agent.get_statistics())
        add_scalar(self.writer, phase, smooth_results, self.train_i_episode)
        self.train_collector.save_infos(phase, self.train_i_episode, self.parser_args.out_result,
                                        self.training_config["save_train_result_n"])

        if self.train_i_episode % self.training_config["save_model_every_n"] == 0:
            self.agent.save("{}/model_epi_{}".format(self.parser_args.out_model, self.train_i_episode))
        print("Episode takes time:{}".format(time.time() - start_time))
        print('Complete training episode {}'.format(self.train_i_episode))

    def evaluate_once(self, epsilon_greedy=False, epsilon=0.15):
        phase = "ZEvaluation"
        start_time = time.time()
        self.test_i_episode += 1
        state, _ = self.env.reset()

        done = False
        infos = []
        i_step = 0
        while not done:
            # choose the policy
            if self.parser_args.policy == Policies.RL_Policy:
                with self.agent.eval_mode():
                    if epsilon_greedy and random.random() < epsilon:
                        action = np.random.randint(0, self.action_space.n)
                    else:
                        action = self.agent.act(state)

                    state, reward, done, info = self.env.step(action)
                    self.agent.observe(obs=state, reward=reward, done=done, reset=False)

            else:
                action = self.agent.act()
                state, reward, done, info = self.env.step(action)

            self.global_i_step += 1
            i_step += 1
            infos.append(info)

        if self.parser_args.policy == Policies.RL_Policy:
            self.test_collector.add(infos)
            self.test_collector.store_plant_types(self.env.plant_types)
            smooth_results = self.test_collector.get_p3d_smooth_statistic(self.env, self.agent.get_statistics())
            add_scalar(self.writer, phase, smooth_results, self.test_i_episode)
            self.test_collector.save_infos(phase, self.test_i_episode, self.parser_args.out_result,
                                           self.training_config["save_test_result_n"])
        else:
            self.test_collector.add(infos)
            self.test_collector.store_plant_types(self.env.plant_types)
            smooth_results = self.test_collector.get_p3d_smooth_statistic(self.env, None)
            add_scalar(self.writer, phase, smooth_results, self.test_i_episode)
            self.test_collector.save_infos(phase, self.test_i_episode, self.parser_args.out_result,
                                           self.training_config["save_test_result_n"])
        print("Episode takes time:{}".format(time.time() - start_time))
        print('Complete evaluation episode {}'.format(self.test_i_episode))

    def evaluate_n_times(self, epsilon_greedy=False, epsilon=0.15, n=10):
        for i in range(n):
            print("\n=====================================Episode:{}=====================================".format(i))
            self.evaluate_once(epsilon_greedy, epsilon)
