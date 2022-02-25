#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/11/22 10:55 AM 
    @Description    :
        
===========================================
"""
import logging
import time
from typing import Dict, List

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from direct.stdpy import threading

from trainer.trainer_helper import save_episodes_info, add_scalar
from utilities.info import EpisodeInfo, InfoCollector


class RosTrainer(object):
    def __init__(self, env, agent, action_space, parser_args):
        self.parser_args = parser_args
        self.training_config = parser_args.training_config
        self.writer = SummaryWriter(log_dir=parser_args.out_board)
        self.env = env
        self.agent = agent
        self.action_space = action_space

        self.train_i_episode = 0
        self.test_i_episode = 0
        self.global_i_step = 0
        self.train_collector = InfoCollector(self.training_config["train_smooth_n"])
        self.test_collector = InfoCollector(self.training_config["test_smooth_n"])
        self.train_step_collector = {}
        self.test_step_collector = {}

        # discrete
        if not parser_args.train or parser_args.resume:
            logging.info("load model from {} {}".format(self.parser_args.in_model, parser_args.in_model_index))
            self.agent.load("{}/model_epi_{}".format(self.parser_args.in_model, parser_args.in_model_index))

    def run(self):
        print("========================================Start running========================================")
        head = self.parser_args.head
        if self.parser_args.train and not head:
            print("Start training")
            for i in range(self.training_config["num_episodes"]):
                print("\nEpisode:{}".format(i))
                self.training()
                if (i + 1) % 10 == 0:
                    print("\nTest Episode:{}".format(i))
                    self.evaluating()
        else:
            print("Start evaluating without head")
            self.evaluate_n_times()

    def training(self):
        phase = "Train"
        start_time = time.time()
        self.train_i_episode += 1
        self.train_step_collector = {}
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
            print_step_info(self.train_i_episode, i_step, info, self.train_step_collector)

        self.train_collector.add(infos)
        smooth_results = self.train_collector.get_ros_smooth_statistic(self.agent.get_statistics())
        self.train_collector.save_infos(phase, self.train_i_episode, self.parser_args.out_result,
                                        self.training_config["save_train_result_n"])
        add_scalar(self.writer, phase, smooth_results, self.train_i_episode)

        if self.train_i_episode % self.training_config["save_model_every_n"] == 0:
            self.agent.save("{}/model_epi_{}".format(self.parser_args.out_model, self.train_i_episode))

        print("Episode takes time:{}".format(time.time() - start_time))
        print('Complete training episode {}'.format(self.train_i_episode))

    def evaluating(self):
        phase = "ZEvaluation"
        start_time = time.time()
        self.test_i_episode += 1
        self.test_step_collector = {}
        state, _ = self.env.reset()

        done = False
        infos = []
        i_step = 0
        with self.agent.eval_mode():
            while not done:
                action = self.agent.act(state)
                state, reward, done, info = self.env.step(action)
                self.agent.observe(obs=state, reward=reward, done=done, reset=False)
                self.global_i_step += 1
                i_step += 1
                infos.append(info)
                print_step_info(self.test_i_episode, i_step, info, self.test_step_collector)

        self.test_collector.add(infos)
        smooth_results = self.test_collector.get_ros_smooth_statistic(self.agent.get_statistics())
        self.test_collector.save_infos(phase, self.test_i_episode, self.parser_args.out_result,
                                       self.training_config["save_test_result_n"])
        add_scalar(self.writer, phase, smooth_results, self.test_i_episode)

        print("Episode takes time:{}".format(time.time() - start_time))
        print('Complete evaluation episode {}'.format(self.test_i_episode))

    def evaluate_n_times(self, n=10):
        for i in range(n):
            print("\nEpisode:{}".format(i))
            self.evaluating()


def print_step_info(episode: int, step: int, info: Dict, step_collector: Dict):
    for key in info.keys():
        if key not in step_collector.keys():
            step_collector[key] = info[key]
        else:
            step_collector[key] += info[key]
    print("Episode : {}; step : {}".format(episode, step))
    print("Found cells:\n  {}".format(info))
    print("Accumulated values:\n  {}\n".format(step_collector))
