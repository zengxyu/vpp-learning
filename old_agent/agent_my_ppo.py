import os

from memory.memory_ppo import MemoryPPO
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn.functional as F


class Agent:
    """description of class"""

    def __init__(self, params, summary_writer, is_resume=False, filepath="", normalize=True):
        self.name = "PPOng"

        self.summary_writer = summary_writer
        # self.frame_size = np.product(field.shape)
        self.pose_size = 2
        self.action_size = params['action_size']
        self.robot_pose_size = params['robot_pose_size']

        self.batch_size = params['batch_size']
        self.seq_len = params['seq_len']
        self.lr = params['lr']
        self.Model = params['model']
        self.EPSILON = 0.2
        self.K_epoch = 16
        self.gamma = params['gamma']

        self.train_device = torch.device('cuda') if torch.cuda.is_available() and params["use_gpu"] else torch.device(
            'cpu')  # 'cuda' if torch.cuda.is_available() else 'cpu'
        print("train device : {}".format(self.train_device))

        print("Action size : {}".format(self.action_size))
        print("Train device : {}".format(self.train_device))
        self.policy = self.get_model(is_resume, filepath)
        print("K epoch : {}".format(self.K_epoch))
        print("Learning rate : {}".format(self.lr))
        print("Batch size : {}".format(self.batch_size))
        print("Seq length : {}".format(self.seq_len))
        print("Gamma : {}".format(self.gamma))

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.memory = MemoryPPO(batch_size=self.batch_size, seq_len=self.seq_len)

        self.frames, self.robot_poses, self.actions, self.rewards, self.frames_prime, self.robot_poses_prime, self.values, self.probs, self.dones = [], [], [], [], [], [], [], [], [],
        self.deltas, self.returns, self.advantages = [], [], []

    def get_model(self, is_resume, filepath):
        policy = self.Model(self.action_size, self.robot_pose_size).to(self.train_device)
        print("Model:", policy)
        if is_resume:
            self.load_model(filename=filepath, map_location=self.train_device)
        return policy

    def load_model(self, filename, map_location=None):  # pass 'cpu' as map location if no cuda available
        state_dict = torch.load(filename, map_location)
        self.policy.load_state_dict(state_dict)

    def store_model(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def store_data(self, transition):
        self.frames.append(transition[0])
        self.robot_poses.append(transition[1])
        self.actions.append(transition[2])
        self.rewards.append(transition[3])
        self.frames_prime.append(transition[4])
        self.robot_poses_prime.append(transition[5])
        self.values.append(transition[6])
        self.probs.append(transition[7])
        self.dones.append(transition[8])

        if len(self.frames) > self.seq_len:
            # compute returns and advantages
            for i in range(0, self.seq_len):
                return_i = self.rewards[i] + self.gamma * self.values[i + 1] * (1 - self.dones[i])
                self.returns.append(return_i)
                self.deltas.append(return_i - self.values[i])

            advantage = 0.0
            for delta in self.deltas[::-1]:
                advantage = self.gamma * advantage + delta
                self.advantages.append(advantage)
            self.advantages.reverse()
            n = self.seq_len
            self.memory.put_data(
                [self.frames[:n], self.robot_poses[:n], self.actions[:n], self.rewards[:n], self.frames_prime[:n],
                 self.robot_poses_prime[:n], self.probs[:n], self.returns[:n], self.advantages[:n], self.dones[:n]])

            self.clear_first_n_elements(n)

        if transition[8]:
            self.clear_all_elements()

    def clear_first_n_elements(self, n):
        self.frames = self.frames[n:]
        self.robot_poses = self.robot_poses[n:]
        self.actions = self.actions[n:]
        self.rewards = self.rewards[n:]
        self.frames_prime = self.frames_prime[n:]
        self.robot_poses_prime = self.robot_poses_prime[n:]
        self.values = self.values[n:]
        self.probs = self.probs[n:]
        self.dones = self.dones[n:]

        self.deltas = self.deltas[n:]
        self.returns = self.returns[n:]
        self.advantages = self.advantages[n:]

    def clear_all_elements(self):
        self.frames, self.robot_poses, self.actions, self.rewards, self.frames_prime, self.robot_poses_prime, self.values, self.probs, self.dones = [], [], [], [], [], [], [], [], [],
        self.deltas, self.returns, self.advantages = [], [], []
        self.h_ins, self.c_ins, self.h_outs, self.c_outs = [], [], [], []

    def train_net(self):
        # print("train net")
        loss_values = []
        loss_clip_values = []
        loss_val_values = []
        loss_ent_values = []
        for i in range(self.K_epoch):
            frames, robot_poses, actions, rewards, frames_prime, robot_poses_prime, old_probs, returns, advantages, dones = self.memory.make_batch(
                self.train_device)

            new_vals, new_probs = self.policy(frames, robot_poses)

            new_prob_a = new_probs.gather(1, actions)
            old_prob_a = old_probs.gather(1, actions)

            ratio = torch.exp(torch.log(new_prob_a) - torch.log(old_prob_a))  # a/b == log(exp(a)-exp(b))

            loss_clip = -torch.min(ratio * advantages,
                                   torch.clamp(ratio, 1 - self.EPSILON, 1 + self.EPSILON) * advantages).mean()
            c1, c2 = 1, 0.01

            loss_val = c1 * F.mse_loss(new_vals, returns)

            loss_ent = -c2 * torch.sum(-torch.log2(new_probs) * new_probs, dim=1).mean()

            self.optimizer.zero_grad()

            loss = loss_clip + loss_val + loss_ent

            loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5)

            self.optimizer.step()

            loss_values.append(loss.item())

            loss_clip_values.append(loss_clip.item())

            loss_val_values.append(loss_val.item())

            loss_ent_values.append(loss_ent.item())

        self.summary_writer.add_loss(np.mean(loss_values))
        self.summary_writer.add_3_loss(np.mean(loss_clip_values), np.mean(loss_val_values),
                                       np.mean(loss_ent_values), np.mean(loss_values))

    def act(self, frame, robot_pose):
        # frame : [seq_len, batch_size, dim, h, w]
        frame_in = torch.Tensor([frame]).to(self.train_device)
        robot_pose_in = torch.Tensor([robot_pose]).to(self.train_device)

        value, probs = self.policy(frame_in, robot_pose_in)
        # probs = torch.clamp(probs, 0, 1)
        if torch.sum(torch.isnan(probs)).item() > 0:
            print('nan values:', torch.sum(torch.isnan(probs)).item())
            for parameters in self.policy.parameters():
                print(parameters)
        # print('Negative values:', torch.sum(probs < 0).item())
        # print("probs shape:", probs.shape)
        # print("probs:", probs)
        categorical = torch.distributions.Categorical(probs)
        action = categorical.sample()
        return action, value, probs

    def get_name(self):
        return self.name
