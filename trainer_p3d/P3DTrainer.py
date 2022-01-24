import logging
import os
import time

import numpy as np

from configs.config import read_yaml
from utilities.Info import EpisodeInfo, StepInfo
from utilities.util import get_project_path

env_config = read_yaml(config_dir=os.path.join(get_project_path(), "configs"), config_name="env.yaml")
headless = env_config["headless"]
if not headless:
    from direct.stdpy import threading


class P3DTrainer(object):
    def __init__(self, env, agent, action_space, writer, parser_config, training_config):
        self.training_config = training_config
        self.env = env
        self.agent = agent
        self.action_space = action_space
        self.writer = writer
        self.render = parser_config.render
        self.train_i_episode = 0
        self.train_i_step = 0
        self.test_i_episode = 0
        self.test_i_step = 0
        self.n_smooth = 200
        self.global_i_step = 0
        self.start_time = time.time()
        self.train_episode_info = EpisodeInfo()
        self.test_episode_info = EpisodeInfo()
        # self.config = config
        # self.summary_writer = SummaryWriterLogger(config)
        # self.logger = BasicLogger.setup_console_logging(config)
        # discrete
        if not parser_config.train:
            agent.turn_off_exploration = True
            agent.load_model(parser_config.in_model_index)

    def run(self):
        if headless:
            for i in range(self.training_config["num_episodes_to_run"]):
                print("\nEpisode:{}".format(i))
                self.training()
                if (i + 1) % 10 == 0:
                    print("\nTest Episode:{}".format(i))
                    self.evaluating()
        else:
            # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
            # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
            main_thread = threading.Thread(target=self.evaluating)
            main_thread.start()
            self.env.gui.run()

    def training(self):
        phase = "Train"
        self.train_i_episode += 1
        state, _ = self.env.reset()

        done = False
        rewards = []
        while not done:
            action = self.agent.act(state)
            state, reward, done, step_info = self.env.step(action)
            self.agent.observe(obs=state, reward=reward, done=done, reset=False)
            self.train_i_step += 1
            self.global_i_step += 1
            rewards.append(reward)

        reward_mean = np.sum(rewards)
        print("reward_sum : {}".format(reward_mean))
        self.train_episode_info.add({"reward_sum": reward_mean})
        add_scalar(self.writer, phase, self.train_episode_info.statistic(), self.train_i_episode)
        print('Complete training episode {}'.format(self.train_i_episode))

    def evaluating(self):
        phase = "ZEvaluation"
        self.test_i_episode += 1
        state, _ = self.env.reset()

        done = False
        rewards = []
        with self.agent.eval_mode():
            while not done:
                action = self.agent.act(state)
                state, reward, done, step_info = self.env.step(action)
                self.agent.observe(obs=state, reward=reward, done=done, reset=False)
                self.train_i_step += 1
                self.global_i_step += 1
                rewards.append(reward)

        reward_mean = np.sum(rewards)
        print("reward_sum : {}".format(reward_mean))
        self.test_episode_info.add({"reward_sum": reward_mean})
        add_scalar(self.writer, phase, self.test_episode_info.statistic(), self.test_i_episode)

        print('Complete evaluation episode {}'.format(self.test_i_episode))


def add_scalar(writer, phase, episode_info, i_episode):
    for key, item in episode_info.items():
        writer.add_scalar(str(phase) + "/" + str(key), item, i_episode)
