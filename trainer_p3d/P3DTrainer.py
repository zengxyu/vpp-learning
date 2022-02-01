import os
import pickle
import time

from config import read_yaml
from utilities.Info import EpisodeInfo
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
        self.train_collector = EpisodeInfo(training_config["smooth_n"])
        self.test_collector = EpisodeInfo(training_config["smooth_n"])
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
        infos = []
        while not done:
            action = self.agent.act(state)
            state, reward, done, info = self.env.step(action)
            self.agent.observe(obs=state, reward=reward, done=done, reset=False)
            self.train_i_step += 1
            self.global_i_step += 1
            infos.append(info)

        add_statistics_to_collector(infos=infos, agent_statistics=self.agent.get_statistics(),
                                    episode_info_collector=self.train_collector, env=self.env)
        add_scalar(self.writer, phase, self.train_collector.get_smooth_statistics(), self.train_i_episode)
        save_episodes_info(phase, self.train_collector, self.training_config, self.train_i_episode)
        if self.train_i_episode % self.training_config["save_model_every_n"] == 0:
            self.agent.save("{}/model_epi_{}".format(self.training_config["out_model"], self.train_i_episode))
        print('Complete training episode {}'.format(self.train_i_episode))

    def evaluating(self):
        phase = "ZEvaluation"
        self.test_i_episode += 1
        state, _ = self.env.reset()

        done = False
        infos = []

        with self.agent.eval_mode():
            while not done:
                action = self.agent.act(state)
                state, reward, done, info = self.env.step(action)
                self.agent.observe(obs=state, reward=reward, done=done, reset=False)
                self.train_i_step += 1
                self.global_i_step += 1
                infos.append(info)

        add_statistics_to_collector(infos=infos, agent_statistics=self.agent.get_statistics(),
                                    episode_info_collector=self.test_collector, env=self.env)
        add_scalar(self.writer, phase, self.test_collector.get_smooth_statistics(), self.test_i_episode)
        save_episodes_info(phase, self.test_collector, self.training_config, self.train_i_episode)
        print('Complete evaluation episode {}'.format(self.test_i_episode))


def add_scalar(writer, phase, episode_info, i_episode):
    for key, item in episode_info.items():
        writer.add_scalar(str(phase) + "/" + str(key), item, i_episode)


def add_statistics_to_collector(infos, agent_statistics, episode_info_collector, env):
    # calculate the statistic info for each episode, then added to episode_info_collector
    new_found_targets_sum = 0
    new_free_cells_sum = 0
    rewards_sum = 0

    for info in infos:
        new_found_targets_sum += info["new_found_targets"]
        new_free_cells_sum += info["new_free_cells"]
        rewards_sum += info["reward"]

    print("rewards_sum : ", rewards_sum)
    episode_info_collector.add({"rewards_sum": rewards_sum})
    episode_info_collector.add({"new_found_targets_sum": new_found_targets_sum})
    episode_info_collector.add({"new_free_cells_sum": new_free_cells_sum})
    episode_info_collector.add({"new_found_targets_rate": new_found_targets_sum / env.target_count})
    episode_info_collector.add({"new_free_cells_rate": new_free_cells_sum / env.free_count})
    episode_info_collector.add({"average_q": agent_statistics[0][1]})
    episode_info_collector.add({"loss": agent_statistics[1][1]})


def save_episodes_info(phase, episode_info_collector, train_config, i_episode):
    save_path = os.path.join(train_config["out_result"], phase + "_log.pkl")
    save_n = train_config["save_result_n"]
    if i_episode % save_n == 0:
        file = open(save_path, 'wb')
        pickle.dump(episode_info_collector.episode_infos, file)
