import os

import numpy as np
import pickle

from torch.utils.tensorboard import SummaryWriter

from utilities.basic_logger import BasicLogger


# from ray import tune


class MySummaryWriter:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

        self.i_update = 0

        self.i_episode = 0

        # the list of mean loss of each episode
        self.mean_losses_list = []
        # losses of each time step in current episode
        self.losses_list = []

        self.sum_rewards_list = []
        self.rewards_list = []

        self.loss_index = 0

    def write_to_summary_board(self):
        self.mean_losses_list.append(np.mean(self.losses_list))
        self.sum_rewards_list.append(np.sum(self.rewards_list))

        self.losses_list = []
        self.rewards_list = []

        self.writer.add_scalar('for_each_episode/l_mean_loss_for_each_episode', self.mean_losses_list[-1],
                               self.i_episode)
        self.writer.add_scalar('for_each_episode/r_mean_reward_for_each_episode', self.sum_rewards_list[-1],
                               self.i_episode)

        # print("Mean loss for episode {} : {}".format(self.i_episode, self.mean_losses_list[-1]))
        # print("Sum reward1 for episode {} : {}".format(self.i_episode, self.sum_rewards_list[-1]))

        self.writer.add_scalar('for_each_episode/l_smoothed_loss',
                               np.mean(self.mean_losses_list[max(0, self.i_episode - 200):]),
                               self.i_episode)
        self.writer.add_scalar('for_each_episode/r_smoothed_reward',
                               np.mean(self.sum_rewards_list[max(0, self.i_episode - 200):]),
                               self.i_episode)

    def add_loss(self, loss):
        self.losses_list.append(loss)

    def add_reward(self, reward, cur_episode):
        if cur_episode == self.i_episode:
            self.rewards_list.append(reward)
        else:
            self.write_to_summary_board()
            self.i_episode = cur_episode
            self.rewards_list.append(reward)

    def add_episode_len(self, episode_len, i_episode):
        self.writer.add_scalar('for_each_episode/e_episode_len', episode_len, i_episode)

    def add_3_loss(self, a_loss, v_loss, ent_loss, loss):
        self.writer.add_scalar('for_each_update/ll_loss_a', a_loss, self.loss_index)
        self.writer.add_scalar('for_each_update/ll_loss_v', v_loss, self.loss_index)
        self.writer.add_scalar('for_each_update/ll_loss_ent', ent_loss, self.loss_index)
        self.writer.add_scalar('for_each_update/ll_loss', loss, self.loss_index)

        self.loss_index += 1

    def add_distance(self, distance, i_episode):
        self.writer.add_scalar('for_each_episode/episode_distance', distance, i_episode)


class SummaryWriterLogger:
    def __init__(self, config):
        self.tb_log_sv_dir = config.folder['tb_log_sv']
        self.tb_l_r_sv_dir = config.folder['tb_l_r_sv']
        self.tb_l_r_in_dir = config.folder['tb_l_r_in']
        self.tb_save_l_r_every_n_episode = config.tb_save_l_r_every_n_episode
        self.tb_smooth_l_r_every_n_episode = config.tb_smooth_l_r_every_n_episode

        assert self.tb_log_sv_dir is not None, "Please input path to saved log..."
        assert self.tb_l_r_sv_dir is not None, "Please input path to saved loss and reward record..."
        assert self.tb_l_r_in_dir is not None, "Please input path to save loss and reward record..."
        self.writer = SummaryWriter(log_dir=self.tb_log_sv_dir)
        self.logger = BasicLogger.setup_console_logging(config)

        self.ep_loss_ll = []
        self.ep_reward_ll = []

        self.ep_inference_loss_ll = []
        self.ep_inference_reward_ll = []

        self.init(self.tb_l_r_in_dir)
        self.show_config(config)

    def init(self, verbose=True):
        """load the file from local"""
        if not os.path.exists(self.tb_l_r_in_dir):
            return
        else:
            print("Load the file from {}".format(self.tb_l_r_in_dir))
            file = open(os.path.join(self.tb_l_r_in_dir, "loss_reward.obj"), 'rb')
            ep_losses, ep_rewards = pickle.load(file)

            for i in range(len(ep_losses)):
                self.update(ep_losses[i], ep_rewards[i], i, verbose=verbose)
            print("Loading done!")

    def show_config(self, config):
        """
        display the configs by text in tensorboard
        """
        count = 0
        for key, value in config.__dict__.items():
            string = str(key) + ":" + str(value)
            print(string)

            self.writer.add_text('Info', string, count)
            count += 1

    def update(self, ep_loss, ep_reward, i_episode, verbose=True):
        """
        # ep_loss : mean loss of this episode
        # ep_reward : sum reward of this episode
        return : mean ep_loss of last n episodes
                mean ep_reward of last n episodes
        """
        self.ep_loss_ll.append(ep_loss)
        self.ep_reward_ll.append(ep_reward)

        self.writer.add_scalar('trainer_p3d/ep_loss', ep_loss, i_episode)
        self.writer.add_scalar('trainer_p3d/ep_reward', ep_reward, i_episode)

        if verbose:
            print("Episode : {} | Mean loss : {} | Reward : {}".format(i_episode, ep_loss, ep_reward))

        ep_loss_smoothed = np.mean(self.ep_loss_ll[max(0, i_episode - self.tb_smooth_l_r_every_n_episode):])
        ep_reward_smoothed = np.mean(self.ep_reward_ll[max(0, i_episode - self.tb_smooth_l_r_every_n_episode):])
        self.writer.add_scalar('trainer_p3d/ep_loss_smoothed', ep_loss_smoothed, i_episode)
        self.writer.add_scalar('trainer_p3d/ep_reward_smoothed', ep_reward_smoothed, i_episode)
        # tune.report(ep_loss_smoothed=ep_loss_smoothed, ep_reward_smoothed=ep_reward_smoothed)
        if i_episode % self.tb_save_l_r_every_n_episode == 0:
            self.save_loss_reward()

        mean_loss_last_n_ep = np.mean(self.ep_loss_ll[max(0, i_episode - self.tb_smooth_l_r_every_n_episode):])
        mean_reward_last_n_ep = np.mean(self.ep_reward_ll[max(0, i_episode - self.tb_smooth_l_r_every_n_episode):])

        return mean_loss_last_n_ep, mean_reward_last_n_ep

    def update_inference_data(self, ep_loss, ep_reward, i_episode, verbose=True):
        """
        # ep_loss : mean loss of this episode
        # ep_reward : sum reward of this episode
        return : mean ep_loss of last n episodes
                mean ep_reward of last n episodes
        """
        self.ep_inference_loss_ll.append(ep_loss)
        self.ep_inference_reward_ll.append(ep_reward)

        self.writer.add_scalar('tester_p3d/ep_loss', ep_loss, i_episode)
        self.writer.add_scalar('tester_p3d/ep_reward', ep_reward, i_episode)

        if verbose:
            print("Episode : {} | Mean loss : {} | Reward : {}".format(i_episode, ep_loss, ep_reward))

        ep_loss_smoothed = np.mean(self.ep_inference_loss_ll[max(0, i_episode - self.tb_smooth_l_r_every_n_episode):])
        ep_reward_smoothed = np.mean(
            self.ep_inference_reward_ll[max(0, i_episode - self.tb_smooth_l_r_every_n_episode):])
        self.writer.add_scalar('tester_p3d/ep_loss_smoothed', ep_loss_smoothed, i_episode)
        self.writer.add_scalar('tester_p3d/ep_reward_smoothed', ep_reward_smoothed, i_episode)

        if i_episode % self.tb_save_l_r_every_n_episode == 0:
            self.save_loss_reward()

        mean_loss_last_n_ep = np.mean(
            self.ep_inference_loss_ll[max(0, i_episode - self.tb_smooth_l_r_every_n_episode):])
        mean_reward_last_n_ep = np.mean(
            self.ep_inference_reward_ll[max(0, i_episode - self.tb_smooth_l_r_every_n_episode):])

        return mean_loss_last_n_ep, mean_reward_last_n_ep

    def save_loss_reward(self):
        """save ep_loss_ll and ep_reward_ll"""
        print("Save ep_loss_ll and ep_reward_ll to {} !".format(self.tb_l_r_sv_dir))
        file = open(os.path.join(self.tb_l_r_sv_dir, "loss_reward.obj"), 'wb')
        obj = [self.ep_loss_ll, self.ep_reward_ll]
        pickle.dump(obj, file)

    def save_inference_loss_reward(self):
        """save ep_loss_ll and ep_reward_ll"""
        print("Save ep_inference_loss_ll and ep_inference_reward_ll to {} !".format(self.tb_l_r_sv_dir))
        file = open(os.path.join(self.tb_l_r_sv_dir, "inference_loss_reward.obj"), 'wb')
        obj = [self.ep_inference_loss_ll, self.ep_inference_reward_ll]
        pickle.dump(obj, file)
