import os

import numpy as np
import pickle

from torch.utils.tensorboard import SummaryWriter


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
    def __init__(self, log_sv_dir, lr_sv_dir, lr_in_dir="", save_every=1, smooth_every=200):
        assert log_sv_dir is not None, "Please input path to save log..."
        assert lr_sv_dir is not None, "Please input path to save loss and reward record..."
        self.writer = SummaryWriter(log_dir=log_sv_dir)
        self.lr_sv_path = lr_sv_dir

        self.save_every = save_every
        self.smooth_every = smooth_every

        self.ep_loss_ll = []
        self.ep_reward_ll = []

        self.init(lr_in_dir)

    def init(self, lr_in_pth, verbose=True):
        """load the file from local"""
        if not os.path.exists(lr_in_pth):
            return
        else:
            print("Load the file from {}".format(lr_in_pth))
            file = open(os.path.join(lr_in_pth, "loss_reward.obj"), 'rb')
            ep_losses, ep_rewards = pickle.load(file)

            for i in range(len(ep_losses)):
                self.update(ep_losses[i], ep_rewards[i], i, verbose=verbose)
            print("Loading done!")

    def update(self, ep_loss, ep_reward, i_episode, verbose=True):
        """
        # ep_loss : mean loss of this episode
        # ep_reward : sum reward of this episode
        """
        self.ep_loss_ll.append(ep_loss)
        self.ep_reward_ll.append(ep_reward)

        self.writer.add_scalar('train/ep_loss', ep_loss, i_episode)
        self.writer.add_scalar('train/ep_reward', ep_reward, i_episode)

        if verbose:
            print("Episode : {} | Mean loss : {} | Reward : {}".format(i_episode, ep_loss, ep_reward))

        self.writer.add_scalar('train/ep_loss_smoothed',
                               np.mean(self.ep_loss_ll[max(0, i_episode - self.smooth_every):]), i_episode)
        self.writer.add_scalar('train/ep_reward_smoothed',
                               np.mean(self.ep_reward_ll[max(0, i_episode - self.smooth_every):]), i_episode)

        if i_episode % self.save_every == 0:
            self.save()

    def save(self):
        """save ep_loss_ll and ep_reward_ll"""
        print("Save ep_loss_ll and ep_reward_ll to {} !".format(self.lr_sv_path))
        file = open(os.path.join(self.lr_sv_path, "loss_reward.obj"), 'wb')
        obj = [self.ep_loss_ll, self.ep_reward_ll]
        pickle.dump(obj, file)
