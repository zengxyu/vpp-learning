import logging
import os
import pickle
import time
from typing import Dict, List

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utilities.Info import EpisodeInfo
from direct.stdpy import threading


class P3DTrainer(object):
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
        self.train_collector = EpisodeInfo(self.training_config["train_smooth_n"])
        self.test_collector = EpisodeInfo(self.training_config["test_smooth_n"])
        # discrete
        if not parser_args.train:
            logging.info("load model from {} {}".format(self.parser_args.in_model, parser_args.in_model_index))
            self.agent.load("{}/model_epi_{}".format(self.parser_args.in_model, parser_args.in_model_index))

    def run(self):
        print("========================================Start running========================================")
        head = self.parser_args.head
        if self.parser_args.train and not head:
            print("Start training")
            for i in range(self.training_config["num_episodes_to_run"]):
                print("\nEpisode:{}".format(i))
                self.training()
                if (i + 1) % 10 == 0:
                    print("\nTest Episode:{}".format(i))
                    self.evaluating()
        else:
            if head:
                print("Start evaluating with head")
                main_thread = threading.Thread(target=self.evaluate_n_times)
                main_thread.start()
                self.env.gui.run()
            else:
                print("Start evaluating without head")
                self.evaluate_n_times()

    def training(self):
        phase = "Train"
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

        add_statistics_to_collector(infos=infos, agent_statistics=self.agent.get_statistics(),
                                    episode_info_collector=self.train_collector, env=self.env)
        add_scalar(self.writer, phase, self.train_collector.get_smooth_statistics(), self.train_i_episode)
        save_episodes_info(phase, self.train_collector, self.train_i_episode, self.parser_args)

        if self.train_i_episode % self.training_config["save_model_every_n"] == 0:
            self.agent.save("{}/model_epi_{}".format(self.parser_args.out_model, self.train_i_episode))

        print('Complete training episode {}'.format(self.train_i_episode))

    def evaluating(self):
        phase = "ZEvaluation"
        self.test_i_episode += 1
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
        add_statistics_to_collector(infos=infos, agent_statistics=self.agent.get_statistics(),
                                    episode_info_collector=self.test_collector, env=self.env)
        add_scalar(self.writer, phase, self.test_collector.get_smooth_statistics(), self.test_i_episode)
        save_episodes_info(phase, self.test_collector, self.test_i_episode, self.parser_args)
        print('Complete evaluation episode {}'.format(self.test_i_episode))

    def evaluate_n_times(self, n=10):
        for i in range(n):
            print("\nEpisode:{}".format(i))
            self.evaluating()


def add_scalar(writer, phase, episode_info, i_episode):
    for key, item in episode_info.items():
        writer.add_scalar(str(phase) + "/" + str(key), item, i_episode)


def print_info():
    pass


def add_statistics_to_collector(infos: List[Dict], agent_statistics, episode_info_collector: EpisodeInfo, env):
    # calculate the statistic info for each episode, then added to episode_info_collector
    new_found_targets_sum = 0
    new_free_cells_sum = 0
    rewards_sum = 0
    visit_gain_sum = 0

    for info in infos:
        visit_gain_sum += info["visit_gain"]
        new_found_targets_sum += info["new_found_targets"]
        new_free_cells_sum += info["new_free_cells"]
        rewards_sum += info["reward"]

    print("rewards_sum : ", rewards_sum)
    print("new_found_targets_sum : ", new_found_targets_sum)
    print("new_free_cells_sum : ", new_free_cells_sum)
    print("visit_gain_sum : ", visit_gain_sum)

    print("new_found_targets_rate : ", new_found_targets_sum / env.target_count)
    print("new_free_cells_rate : ", new_free_cells_sum / env.free_count)
    print("coverage rate : ", infos[-1]["coverage_rate"])

    episode_info_collector.add({"rewards_sum": rewards_sum})
    episode_info_collector.add({"new_found_targets_sum": new_found_targets_sum})
    episode_info_collector.add({"new_free_cells_sum": new_free_cells_sum})
    episode_info_collector.add({"visit_gain_sum": visit_gain_sum})

    episode_info_collector.add({"new_found_targets_rate": new_found_targets_sum / env.target_count})
    episode_info_collector.add({"new_free_cells_rate": new_free_cells_sum / env.free_count})
    episode_info_collector.add({"coverage_rate": infos[-1]["coverage_rate"]})

    if not np.isnan(agent_statistics[0][1]):
        episode_info_collector.add({"average_q": agent_statistics[0][1]})
        episode_info_collector.add({"loss": agent_statistics[1][1]})


def save_episodes_info(phase, episode_info_collector, i_episode, parser_args):
    save_path = os.path.join(parser_args.out_folder, phase + "_log.pkl")
    save_n = parser_args.training_config["save_result_n"]
    if i_episode % save_n == 0:
        file = open(save_path, 'wb')
        pickle.dump(episode_info_collector.episode_infos, file)
