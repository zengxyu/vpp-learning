import random
import os
import torch.nn.functional as F

from memory.replay_buffer_hierarchical import ManagerPriorityReplayBuffer, WorkerPriorityReplayBuffer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch.optim as optim

import numpy as np
import torch

from network.network_dqn import DQN_Network, DQN_Network4


class Agent:
    def __init__(self, params, summary_writer, model_path=""):
        self.name = "grid world"
        self.params = params
        self.update_every = params['update_every']
        self.manager_update_every = params['manager_update_every']
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

        self.WorkerModel = params['model']
        self.ManagerModel = params['manager_model']
        self.policy_net = self.WorkerModel(self.action_size).to(self.device)
        self.target_net = self.WorkerModel(self.action_size).to(self.device)

        if not model_path == "":
            # resume model if model path == ""
            self.resume_model(model_path, model_path)

        if not self.params['is_train']:
            # 如果不是训练状态的话，不更新
            self.update_every = 1000000000000000000

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.manager_memory = ManagerPriorityReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size,
                                                          device=self.device,
                                                          normalizer=None, seed=self.seed)
        self.worker_memory = WorkerPriorityReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size,
                                                        device=self.device,
                                                        normalizer=None, seed=self.seed)

        self.manager_policy_net = self.ManagerModel().to(self.device)
        self.manager_target_net = self.ManagerModel().to(self.device)
        self.manager_actor_optimizer = optim.Adam(self.manager_policy_net.get_actor_params(), lr=1e-4)
        self.manager_critic_optimizer = optim.Adam(self.manager_policy_net.get_critic_params(), lr=1e-4)
        self.thred = 30
        print("device:", self.device)
        print("gamma:", self.gamma)
        print(self.policy_net)
        print(self.manager_policy_net)

        self.time_step = 0
        self.manager_time_step = 0
        self.c_step = 0

    def store_worker_experience(self, state, action, reward, next_state, done):
        # self.memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.worker_memory.add_experience(state=state, action=action, reward=reward, next_state=next_state, done=done)

    def store_manager_experience(self, state, reward, next_state, done):
        self.manager_memory.add_experience(state=state, reward=reward, next_state=next_state, done=done)

    def resume_model(self, model_path, map_location):
        state_dict = torch.load(model_path, map_location=map_location)
        self.policy_net.load_state_dict(state_dict)
        self.update_target_network()

    def store_model(self, file_path):
        torch.save(self.policy_net.state_dict(), file_path)

    def reset(self):
        pass

    def manager_act(self, observed_map, robot_pose):
        observed_map = torch.Tensor([observed_map]).to(self.device)
        robot_pose_in = torch.Tensor([robot_pose]).to(self.device)
        self.manager_policy_net.eval()
        with torch.no_grad():
            action = self.manager_policy_net.policy(observed_map, robot_pose_in)

        action = action.cpu().data.numpy().squeeze()

        return action

    def act(self, robot_pose):
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
            # frame_in = torch.Tensor([frame]).to(self.device)
            robot_pose_in = torch.Tensor([robot_pose]).to(self.device)

            self.policy_net.eval()
            with torch.no_grad():
                q_val = self.policy_net(robot_pose_in)

            action = np.argmax(q_val.cpu().data.numpy())

        return action

    def learn_manager_actor(self, observed_map, robot_pose):
        action = self.manager_policy_net.policy(observed_map, robot_pose)

        Q = self.manager_policy_net.value(observed_map, robot_pose, action)

        self.manager_actor_optimizer.zero_grad()
        cost1 = -1.0 * Q
        cost1_mean = torch.mean(cost1)
        # action 是一个没有被归一化的方向
        cost2 = torch.norm(action[:, :3], dim=1).unsqueeze(1)
        cost2 = torch.where(cost2 < self.thred, torch.zeros_like(cost2), cost2)
        cost2_mean = torch.mean(cost2)
        cost = cost1_mean + 0.01 * cost2_mean

        cost.backward()

        for p in self.manager_policy_net.actor_model.parameters():
            p.grad.data.clamp_(-1, 1)

        self.manager_actor_optimizer.step()
        # print("cost1:", cost1_mean)
        # print("cost2:", cost2_mean)
        return cost1

    def learn_manager_critic(self, observed_map, robot_pose, action, reward, next_observed_map, next_robot_pose,
                             terminal):
        next_action = self.manager_target_net.policy(next_observed_map, next_robot_pose)
        next_Q = self.manager_target_net.value(next_observed_map, next_robot_pose, next_action)
        target_Q = reward + (1 - terminal) * self.gamma * next_Q
        Q = self.manager_policy_net.value(observed_map, robot_pose, action)

        self.manager_critic_optimizer.zero_grad()
        cost1 = F.mse_loss(target_Q, Q)

        # cost2 = torch.norm((robot_pose[:, :3] - 128) / 128 - action[:, :3], dim=1).unsqueeze(1)
        # cost2 = torch.norm(action[:, :3], dim=1).unsqueeze(1)

        # cost2 = torch.where(cost2 < self.thred, torch.zeros_like(cost2), cost2)
        # c2 = 100
        # cost2_mean = torch.mean(cost2)
        cost = cost1
        cost.backward()

        for p in self.manager_policy_net.critic_model.parameters():
            p.grad.data.clamp_(-1, 1)

        print("cost1:", cost1)
        # print("cost2:", cost2_mean)
        # print("cost:", cost)

        self.manager_critic_optimizer.step()

        return torch.abs(target_Q - Q)

    def learn_manager(self):
        self.policy_net.train()
        self.target_net.eval()

        if len(self.manager_memory) > self.batch_size:
            print("learn manager")
            tree_idx, minibatch, ISWeights = self.manager_memory.sample()

            frames_in, robot_poses_in, actions, rewards, next_frames_in, next_robot_poses_in, dones = minibatch
            # print("sampled batch rewards:", rewards)
            # print("sampled batch ISWeights:", ISWeights)

            actor_loss = self.learn_manager_actor(frames_in, robot_poses_in)
            critic_loss = self.learn_manager_critic(frames_in, robot_poses_in, actions, rewards, next_frames_in,
                                                    next_robot_poses_in, dones)
            # rewards = torch.where(rewards < 0, cost2, 0)
            loss_reward_each_item = actor_loss + critic_loss + rewards
            loss_reward_each_item = torch.where(torch.isnan(loss_reward_each_item),
                                                torch.zeros_like(loss_reward_each_item), loss_reward_each_item)
            loss_reward_each_item = loss_reward_each_item.detach().cpu().numpy()

            self.manager_memory.batch_update(tree_idx, loss_reward_each_item)
            self.manager_time_step += 1

        if (self.manager_time_step + 1) % self.manager_update_every == 0:
            print("==================update manager time step=======================")
            self.update_manager_target_network()

    def learn_worker(self):
        self.policy_net.train()
        self.target_net.eval()
        loss_value = 0

        if len(self.worker_memory) > self.batch_size:
            # print("learn worker")
            tree_idx, minibatch, ISWeights = self.worker_memory.sample()

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
            self.worker_memory.batch_update(tree_idx, loss_reward_each_item)
            self.time_step += 1

        if (self.time_step + 1) % self.update_every == 0:
            # print("============update worker==========")
            self.update_target_network()
        return loss_value

    def weighted_mse_loss(self, input, target, weight):
        return torch.sum(torch.from_numpy(weight).to(self.device) * (input - target) ** 2)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_manager_target_network(self):
        self.manager_target_net.actor_model.load_state_dict(self.manager_policy_net.actor_model.state_dict())
        self.manager_target_net.critic_model.load_state_dict(self.manager_policy_net.critic_model.state_dict())
