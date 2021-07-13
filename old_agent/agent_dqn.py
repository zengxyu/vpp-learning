import pickle
import random
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import torch
import torch.optim as optim

from memory.replay_buffer import PriorityReplayBuffer


class Agent:
    def __init__(self, params, summary_writer, model_in_pth="", exp_in_path=""):
        self.name = "grid world"
        self.params = params
        self.update_every = params['update_every']
        self.eps = params['eps_start']
        self.eps_decay = params['eps_decay']
        self.eps_min = params['eps_min']
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.exp_in_path = exp_in_path
        self.seed = random.seed(42)

        self.action_size = params['action_size']
        self.device = torch.device("cuda") if torch.cuda.is_available() and params['use_gpu'] else torch.device("cpu")
        self.summary_writer = summary_writer
        print("device:", self.device)
        print("gamma:", self.gamma)
        # 目标target

        self.Network = params['network']
        self.policy_net = self.Network(self.action_size).to(self.device)
        self.target_net = self.Network(self.action_size).to(self.device)
        if not model_in_pth == "":
            print("resume model")
            self.load_model(model_pth=model_in_pth, map_location=self.device)
            self.update_target_network()
            if not params['is_train']:
                # 如果不是训练状态的话，不更新
                self.update_every = 1000000000000000000
        print(self.policy_net)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        if exp_in_path == "":
            self.memory = PriorityReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size,
                                               device=self.device, seed=self.seed, is_continuous=False)
        else:
            self.memory = pickle.load(open(exp_in_path, 'rb'))
            print("loaded replay buffer size:{}".format(self.memory.size))
        self.time_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add_experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.time_step += 1
        if self.time_step % 100 == 0:
            print("save replay buffer to disk")
            f = open(os.path.join(self.params['exp_sv'], 'buffer.obj'), 'wb')
            pickle.dump(self.memory, f)

    def load_model(self, model_pth, map_location):
        state_dict = torch.load(model_pth, map_location=map_location)
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
        if self.params['is_train']:
            self.eps = max(self.eps * self.eps_decay, self.eps_min)
        else:
            self.eps = 0
        if rnd < self.eps:
            return np.random.randint(self.action_size)
        else:
            frame_in = torch.Tensor([frame]).to(self.device)
            robot_pose_in = torch.Tensor([robot_pose]).to(self.device)

            self.policy_net.eval()
            with torch.no_grad():
                q_val = self.policy_net([frame_in, robot_pose_in])

            action = np.argmax(q_val.cpu().data.numpy())

        return action

    def learn(self):
        self.policy_net.train()
        self.target_net.eval()
        loss_value = 0

        # if len(self.memory) > self.batch_size:
        if True:
            tree_idx, minibatch, ISWeights = self.memory.sample()
            states, actions, rewards, next_states, dones = minibatch
            q_values = self.target_net(next_states).detach()
            max_action_values = q_values.max(1)[0].unsqueeze(1)
            # If done just use reward, else update Q_target with discounted action values
            Q_target = rewards + (self.gamma * max_action_values * (1 - dones))
            Q_action_values = self.policy_net(states)
            Q_expected = Q_action_values.gather(1, actions)

            self.optimizer.zero_grad()
            loss = self.weighted_mse_loss(Q_expected, Q_target, ISWeights)
            loss.backward()

            for p in self.policy_net.parameters():
                p.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            loss_value = loss.item()

            Q_expected2 = self.policy_net(states).gather(1, actions)
            loss_each_item = torch.abs(Q_expected2 - Q_target)

            loss_reward_each_item = loss_each_item + rewards
            loss_reward_each_item = loss_reward_each_item.detach().cpu().numpy()
            tree_idx = tree_idx[:, np.newaxis]
            self.memory.batch_update(tree_idx, loss_reward_each_item)
        if self.time_step % self.update_every == 0:
            self.update_target_network()
        return loss_value

    def weighted_mse_loss(self, input, target, weight):
        return torch.sum(weight * (input - target) ** 2)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
