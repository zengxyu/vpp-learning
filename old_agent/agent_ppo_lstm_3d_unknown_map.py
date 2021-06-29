import os
import pickle

from memory.RewardRecord import RewardRecord
from memory.memory_ppo_lstm import MemoryPPOLSTM

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn.functional as F


class Agent:
    """description of class"""

    def __init__(self, params, summary_writer, is_resume=False, filepath="", normalize=True):
        self.summary_writer = summary_writer
        self.normalize = normalize
        self.pose_size = 2
        self.action_size = params['action_size']
        self.robot_pose_size = params['robot_pose_size']
        self.batch_size = params['batch_size']
        self.seq_len = params['seq_len']
        self.num_layers = params['num_layers']
        self.lr = params['lr']
        self.Model = params['model']
        self.EPSILON = 0.2
        self.K_epoch = 16
        self.gamma = params['gamma']
        self.train_device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
            'cpu')  # 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Action size : {}".format(self.action_size))
        print("Train device : {}".format(self.train_device))
        self.policy = self.get_model(is_resume, filepath)
        print("K epoch : {}".format(self.K_epoch))
        print("Learning rate : {}".format(self.lr))
        print("Batch size : {}".format(self.batch_size))
        print("Seq length : {}".format(self.seq_len))
        print("Num layers : {}".format(self.num_layers))
        print("Gamma : {}".format(self.gamma))

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.9, last_epoch=-1)
        self.memory = MemoryPPOLSTM(batch_size=self.batch_size, seq_len=self.seq_len)

        self.frames, self.robot_poses, self.actions, self.rewards, self.frames_prime, self.robot_poses_prime, self.values, self.probs, self.dones = [], [], [], [], [], [], [], [], [],
        self.deltas, self.returns, self.advantages = [], [], []
        self.h_ins, self.c_ins, self.h_outs, self.c_outs = [], [], [], []

        if self.normalize:
            self.observed_map_mean, self.observed_map_std, self.robot_pose_mean, self.robot_pose_std, self.reward_mean, self.reward_std = self.load_data_mean_std(
                config_dir=params['config_dir'])

            self.reward_record = RewardRecord()

    def load_data_mean_std(self, config_dir):
        with open(os.path.join(config_dir, "observed_map_mean_std.pkl"), 'rb') as f:
            observed_map_mean, observed_map_std = pickle.load(f)

        with open(os.path.join(config_dir, "robot_pose_mean_std.pkl"), 'rb') as f:
            robot_pose_mean, robot_pose_std = pickle.load(f)

        with open(os.path.join(config_dir, "reward_mean_std1.pkl"), 'rb') as f:
            reward_mean, reward_std = pickle.load(f)

        return observed_map_mean, observed_map_std, robot_pose_mean, robot_pose_std, reward_mean, reward_std

    def get_model(self, is_resume, filepath):
        policy = self.Model(self.action_size, self.robot_pose_size, self.num_layers).to(self.train_device)
        print("Model name:", self.Model)
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
        frame = transition[0]
        robot_pose = transition[1]
        frames_prime = transition[4]
        robot_poses_prime = transition[5]
        if self.normalize:
            frame = (frame - self.observed_map_mean) / (self.observed_map_std + 1e-8)
            robot_pose = (robot_pose - self.robot_pose_mean) / (self.robot_pose_std + 1e-8)
            frames_prime = (frames_prime - self.observed_map_mean) / (self.observed_map_std + 1e-8)
            robot_poses_prime = (robot_poses_prime - self.robot_pose_mean) / (self.robot_pose_std + 1e-8)

        self.frames.append(frame)
        self.robot_poses.append(robot_pose)
        self.actions.append(transition[2])
        self.rewards.append(transition[3])
        self.frames_prime.append(frames_prime)
        self.robot_poses_prime.append(robot_poses_prime)
        self.values.append(transition[6])
        self.probs.append(transition[7])
        self.dones.append(transition[8])

        self.h_ins.append(transition[9])
        self.c_ins.append(transition[10])
        self.h_outs.append(transition[11])
        self.c_outs.append(transition[12])

        # print("len(self.frames):", len(self.frames))
        # print("self.seq_len:", self.seq_len)
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
                 self.robot_poses_prime[:n], self.probs[:n], self.returns[:n], self.advantages[:n], self.h_ins[0],
                 self.c_ins[0], self.h_outs[0], self.c_outs[0]])

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

        self.h_ins = self.h_ins[n:]
        self.c_ins = self.c_ins[n:]
        self.h_outs = self.h_outs[n:]
        self.c_outs = self.c_outs[n:]

    def clear_all_elements(self):
        self.frames, self.robot_poses, self.actions, self.rewards, self.frames_prime, self.robot_poses_prime, self.values, self.probs, self.dones = [], [], [], [], [], [], [], [], [],
        self.deltas, self.returns, self.advantages = [], [], []
        self.h_ins, self.c_ins, self.h_outs, self.c_outs = [], [], [], []

    def train_net(self):
        # print("train net")
        loss_v = 0
        for i in range(self.K_epoch):
            frames, robot_poses, actions, rewards, frames_prime, robot_poses_prime, probs, returns, advantages, h_in, c_in, h_out, c_out = self.memory.make_batch(
                self.train_device)

            new_vals, new_probs, _, _ = self.policy(frames, robot_poses, h_in, c_in)
            old_probs = probs.squeeze(2)

            new_prob_a = new_probs.gather(2, actions)
            old_prob_a = old_probs.gather(2, actions)

            ratio = torch.exp(torch.log(new_prob_a) - torch.log(old_prob_a))  # a/b == log(exp(a)-exp(b))

            c1, c2 = 1, 0.01

            loss_clip = -torch.min(ratio * advantages,
                                   torch.clamp(ratio, 1 - self.EPSILON, 1 + self.EPSILON) * advantages).mean()

            loss_mse = c1 * F.mse_loss(new_vals, returns)

            new_probs_reshape = torch.reshape(new_probs, (-1, new_probs.size()[-1]))

            loss_ent = -c2 * torch.sum(-torch.log2(new_probs_reshape) * new_probs_reshape, dim=1).mean()

            self.optimizer.zero_grad()

            loss = loss_clip + loss_mse + loss_ent
            loss.backward(retain_graph=True)
            self.summary_writer.add_3_loss(loss_clip.item(), loss_mse.item(), loss_ent.item(), loss.item())

            self.optimizer.step()

            # print("loss mean:{}".format(loss.mean().item()))
            loss_v += loss.item()

        return loss_v / self.K_epoch

    def act(self, frame, robot_pose, h_in, c_in):
        # frame : [seq_len, batch_size, dim, h, w]
        frame_in = torch.Tensor([[frame]]).to(self.train_device)
        robot_pose_in = torch.Tensor([[robot_pose]]).to(self.train_device)
        h_in, c_in = h_in.to(self.train_device), c_in.to(self.train_device)

        value, probs, h_out, c_out = self.policy(frame_in, robot_pose_in, h_in, c_in)
        categorical = torch.distributions.Categorical(probs)
        action = categorical.sample()
        return action, value, probs, h_out, c_out
