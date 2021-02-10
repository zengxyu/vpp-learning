import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn.functional as F
from src.env.config import ACTION
from src.network.network_ppo_lstm import PPO_LSTM
from src.memory.memory_ppo_lstm import MemoryPPOLSTM


class Agent:
    """description of class"""

    def __init__(self, params, writer, train_agent=False, is_resume=False, filepath="",
                 train_device=torch.device('cuda')):
        self.name = "PPOng"
        self.train_agent = train_agent
        self.writer = writer
        # self.frame_size = np.product(field.shape)
        self.pose_size = 2
        self.action_size = params['action_size']
        self.batch_size = params['batch_size']
        self.seq_len = params['seq_len']
        self.num_layers = params['num_layers']
        self.lr = params['lr']
        self.Model = params['model']
        # self.FRAME_SKIP = 1
        self.EPSILON = 0.2
        self.K_epoch = 16
        self.gamma = params['gamma']
        self.train_device = train_device  # 'cuda' if torch.cuda.is_available() else 'cpu'
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

    def get_model(self, is_resume, filepath):
        policy = self.Model(self.action_size, self.num_layers).to(self.train_device)
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
        self.frames.append(transition[0])
        self.robot_poses.append(transition[1])
        self.actions.append(transition[2])
        self.rewards.append(transition[3])
        self.frames_prime.append(transition[4])
        self.robot_poses_prime.append(transition[5])
        self.values.append(transition[6])
        self.probs.append(transition[7])
        self.dones.append(transition[8])

        self.h_ins.append(transition[9])
        self.c_ins.append(transition[10])
        self.h_outs.append(transition[11])
        self.c_outs.append(transition[12])

        # print("len(self.frames):", len(self.frames))
        # print("self.seq_len:", self.seq_len)
        # len > 0 or an episode finished
        if len(self.frames) > self.seq_len:
            # compute returns and advantages
            for i in range(0, self.seq_len):
                return_i = self.rewards[i] + self.gamma * self.values[i + 1] * self.dones[i]
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
            # new_categorical = torch.distributions.Categorical(new_probs)
            # old_categorical = torch.distributions.Categorical(old_probs)
            new_prob_a = new_probs.gather(2, actions)
            old_prob_a = old_probs.gather(2, actions)

            ratio = torch.exp(torch.log(new_prob_a) - torch.log(old_prob_a))  # a/b == log(exp(a)-exp(b))

            loss_clip = -torch.min(ratio * advantages,
                                   torch.clamp(ratio, 1 - self.EPSILON, 1 + self.EPSILON) * advantages).mean()

            loss_mse = F.mse_loss(new_vals, returns)
            # loss_ent = -new_categorical.entropy().mean()

            c1, c2 = 1, 0.01

            self.optimizer.zero_grad()

            loss = loss_clip + c1 * loss_mse
            loss.backward(retain_graph=True)

            self.optimizer.step()

            # print("loss mean:{}".format(loss.mean().item()))
            loss_v += loss.item()

        return loss_v / self.K_epoch

    def act(self, frame, robot_pose, h_in, c_in):
        # frame : [seq_len, batch_size, dim, h, w]
        frame_in = torch.Tensor([[[frame]]]).to(self.train_device)
        robot_pose_in = torch.Tensor([[robot_pose]]).to(self.train_device)
        h_in, c_in = h_in.to(self.train_device), c_in.to(self.train_device)

        value, probs, h_out, c_out = self.policy(frame_in, robot_pose_in, h_in, c_in)

        categorical = torch.distributions.Categorical(probs)
        action = categorical.sample()
        return action, value, probs, h_out, c_out

    def get_name(self):
        return self.name
