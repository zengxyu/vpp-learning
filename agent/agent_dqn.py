import random
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch.optim as optim

import numpy as np
import torch

from memory.replay_buffer import PriorityReplayBuffer
from network.network_dqn import DQN_Network, DQN_Network4


class Agent:
    def __init__(self, params, summary_writer, model_path=""):
        self.name = "grid world"
        self.params = params
        self.update_every = params['update_every']
        self.eps = params['eps_start']
        self.eps_decay = params['eps_decay']
        self.eps_min = params['eps_min']
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.seed = random.seed(42)

        self.is_normalize = params['is_normalize']

        self.is_double = params['is_double']

        self.action_size = params['action_size']
        self.device = torch.device("cuda") if torch.cuda.is_available() and params['use_gpu'] else torch.device("cpu")
        self.summary_writer = summary_writer
        print("device:", self.device)
        print("gamma:",self.gamma)
        # 目标target

        self.Model = params['model']
        self.policy_net = self.Model(self.action_size).to(self.device)
        self.target_net = self.Model(self.action_size).to(self.device)
        if not model_path == "":
            self.load_model(file_path=model_path, map_location=self.device)
            self.update_target_network()
            if not params['is_train']:
                # 如果不是训练状态的话，不更新
                self.update_every = 1000000000000000000
        print(self.policy_net)
        # self.normalizer = Normalizer(config_dir=params['memory_config_dir']) if self.is_normalize else None

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        # self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-4)
        # if params['is_priority_buffer']:
        self.memory = PriorityReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size,
                                           device=self.device,
                                           normalizer=None, seed=self.seed)

        # else:
        #     self.memory = ReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size, device=self.device,
        #                                seed=self.seed)
        self.time_step = 0

    def save_normalized_memory(self, save_dir):
        print("mean value is saved to memory!")
        self.memory.normalize(save_dir=save_dir)

    def step(self, state, action, reward, next_state, done):
        # self.memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.memory.add_experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.time_step += 1

    def load_model(self, file_path, map_location):
        state_dict = torch.load(file_path, map_location=map_location)
        self.policy_net.load_state_dict(state_dict)

    def store_model(self, file_path):
        torch.save(self.policy_net.state_dict(), file_path)

    def reset(self):
        pass

    def act(self, frame, robot_pose):
        """

        :param frame: [w,h]
        :param robot_pose: [1,2]
        :return:
        """
        rnd = random.random()
        self.eps = max(self.eps * self.eps_decay, self.eps_min)

        if rnd < self.eps:
            return np.random.randint(self.action_size)
        else:
            frame_in = torch.Tensor([frame]).to(self.device)
            robot_pose_in = torch.Tensor([robot_pose]).to(self.device)

            if self.is_normalize:
                frame_in = self.normalizer.normalize_frame_in(frame_in)
                robot_pose_in = self.normalizer.normalize_robot_pose_in(robot_pose_in)

            self.policy_net.eval()
            with torch.no_grad():
                q_val = self.policy_net(robot_pose_in)

            action = np.argmax(q_val.cpu().data.numpy())

        return action

    def learn(self, memory_config_dir):
        self.policy_net.train()
        self.target_net.eval()
        loss_value = 0

        if len(self.memory) > self.batch_size:
            tree_idx, minibatch, ISWeights = self.memory.sample()

            # sampled_experiences = self.memory.sample(self.is_normalize, memory_config_dir)
            # sampled_experiences = self.memory.sample()

            frames_in, robot_poses_in, actions, rewards, next_frames_in, next_robot_poses_in, dones = minibatch

            # Get the action with max Q value
            # frames_in, robot_poses_in = states
            # next_frames_in, next_robot_poses_in = next_states
            if self.is_double:
                q_values = self.policy_net( next_robot_poses_in).detach()
                max_action_next = q_values.max(1)[1].unsqueeze(1)
                Q_target = self.target_net(next_frames_in, next_robot_poses_in).gather(1, max_action_next)
                Q_target = rewards + (self.gamma * Q_target * (1 - dones))
                Q_expected = self.policy_net(frames_in, robot_poses_in).gather(1, actions)
            else:
                q_values = self.target_net( next_robot_poses_in).detach()
                max_action_values = q_values.max(1)[0].unsqueeze(1)
                # If done just use reward, else update Q_target with discounted action values
                Q_target = rewards + (self.gamma * max_action_values * (1 - dones))
                Q_action_values = self.policy_net(robot_poses_in)
                Q_expected = Q_action_values.gather(1, actions)
                # self.update_q_action_values(Q_action_values, robot_poses_in)

            self.optimizer.zero_grad()
            loss = self.weighted_mse_loss(Q_expected, Q_target, ISWeights)
            loss.backward()
            # for name, parms in self.policy_net.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
            for p in self.policy_net.parameters():
                p.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            loss_value = loss.item()

            Q_expected2 = self.policy_net(robot_poses_in).gather(1, actions)
            loss_each_item = torch.abs(Q_expected2 - Q_target)
            rewards[rewards < 0] = 0
            loss_reward_each_item = loss_each_item + rewards
            loss_reward_each_item = loss_reward_each_item.detach().cpu().numpy()
            tree_idx = tree_idx[:, np.newaxis]
            self.memory.batch_update(tree_idx, loss_reward_each_item)
        if self.time_step % self.update_every == 0:
            self.update_target_network()
        return loss_value

    def weighted_mse_loss(self, input, target, weight):
        return torch.sum(torch.from_numpy(weight).to(self.device) * (input - target) ** 2)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())