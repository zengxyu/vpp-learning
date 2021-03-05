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
        self.device = torch.device("cuda") if torch.cuda.is_available() and params['use_gpu'] else torch.device("cpu")

        self.is_double = params['is_double']

        self.action_size = params['action_size']
        self.summary_writer = summary_writer

        # 目标target

        self.Model = params['model']
        self.policy_net = self.Model(self.action_size).to(self.device)
        self.target_net = self.Model(self.action_size).to(self.device)

        if not model_path == "":
            # resume model if model path == ""
            self.resume_model(model_path, model_path)

        if not self.params['is_train']:
            # 如果不是训练状态的话，不更新
            self.update_every = 1000000000000000000

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = PriorityReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size,
                                           device=self.device,
                                           normalizer=None, seed=self.seed)
        print("device:", self.device)
        print("gamma:", self.gamma)
        print(self.policy_net)

        self.time_step = 0
        self.c_step = 0

    def step(self, state, action, reward, next_state, done):
        # self.memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.memory.add_experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.time_step += 1

    def resume_model(self, model_path, map_location):
        state_dict = torch.load(model_path, map_location=map_location)
        self.policy_net.load_state_dict(state_dict)
        self.update_target_network()

    def store_model(self, file_path):
        torch.save(self.policy_net.state_dict(), file_path)

    def reset(self):
        pass

    def act(self, goal, robot_pose):
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

            self.policy_net.eval()
            with torch.no_grad():
                q_val = self.policy_net(robot_pose_in)

            action = np.argmax(q_val.cpu().data.numpy())

        return action

    def learn_manager(self):
        self.policy_net.train()
        self.target_net.eval()


    def learn_worker(self):
        self.policy_net.train()
        self.target_net.eval()
        loss_value = 0

        if len(self.memory) > self.batch_size:
            tree_idx, minibatch, ISWeights = self.memory.sample()

            frames_in, robot_poses_in, actions, rewards, next_frames_in, next_robot_poses_in, dones = minibatch

            # Get the action with max Q value
            # frames_in, robot_poses_in = states
            # next_frames_in, next_robot_poses_in = next_states
            if self.is_double:
                q_values = self.policy_net(next_robot_poses_in).detach()
                max_action_next = q_values.max(1)[1].unsqueeze(1)
                Q_target = self.target_net(next_frames_in, next_robot_poses_in).gather(1, max_action_next)
                Q_target = rewards + (self.gamma * Q_target * (1 - dones))
                Q_expected = self.policy_net(frames_in, robot_poses_in).gather(1, actions)
            else:
                q_values = self.target_net(next_robot_poses_in).detach()
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
