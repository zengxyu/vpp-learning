import numpy as np

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
