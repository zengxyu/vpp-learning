import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from PIL import Image
import random
import copy

from field_env import Field, Action


class PPOPolicy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.con1 = torch.nn.Conv2d(1, 16, kernel_size=16, stride=8)
        self.con2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # self.con3 = torch.nn.Conv2d(32, 32, 3)
        # self.sm = torch.nn.Softmax2d()
        self.fc1 = torch.nn.Linear(128, 32)
        # self.fc2 = torch.nn.Linear(64, 64)
        # self.fc3 = torch.nn.Linear(64, 64)

        self.fc_pose = torch.nn.Linear(3, 32)

        self.fc_val = torch.nn.Linear(64, 1)
        self.fc_pol = torch.nn.Linear(64, action_space)

        # Initialize neural network weights
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, frame, robot_pose):
        out = self.con1(frame)
        out = F.relu(out)
        out = self.con2(out)
        out = F.relu(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)

        out_pose = self.fc_pose(robot_pose)
        out_pose = F.relu(out_pose)
        out = torch.cat((out, out_pose), dim=1)

        val = self.fc_val(out)
        pol = self.fc_pol(out)
        pol = F.softmax(pol, dim=1)
        return val, pol


class Agent:
    """description of class"""

    def __init__(self, field, train_agent=False):
        self.name = "PPOng"
        self.train_agent = train_agent
        self.FRAME_SIZE = np.product(field.shape)
        self.ACTION_SPACE = len(Action)
        self.TRAJ_COLLECTION_NUM = 16
        self.TRAJ_LEN = 4
        # self.BATCH_SIZE = 32
        self.tlen_counter = 0
        self.tcol_counter = 0
        # self.FRAME_SKIP = 1
        self.EPSILON = 0.2
        self.EPOCH_NUM = 4
        self.gamma = 0.98
        self.train_device = torch.device('cuda')  # 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy = PPOPolicy(self.FRAME_SIZE, self.ACTION_SPACE).to(self.train_device)
        # self.old_policy = copy.deepcopy(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)

        self.traj_memory = [[], [], [], [], [], []]

        self.frames = []
        self.robot_poses = []
        self.actions = []
        self.values = []
        self.probs = []
        self.deltas = []
        self.last_frame = None
        self.last_robot_pose = None
        self.last_action = None
        self.last_probs = None
        self.last_reward = None
        self.last_value = None
        # self.observations = []
        # self.values = []
        # self.best_values = []
        # self.rewards = []

    def load_model(self, filename, map_location=None):  # pass 'cpu' as map location if no cuda available
        state_dict = torch.load(filename, map_location)
        self.policy.load_state_dict(state_dict)

    def store_model(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def reset(self):
        self.last_frame = None
        self.last_robot_pose = None
        self.last_action = None
        self.last_probs = None
        self.last_reward = None
        self.last_value = None

    def store_trajectory_to_memory(self):
        T = len(self.deltas)
        advantages = torch.zeros(T, 1).to(self.train_device)
        returns = torch.zeros(T, 1).to(self.train_device)
        advantages[T - 1] = self.deltas[T - 1]
        returns[T - 1] = advantages[T - 1] + self.values[T - 1]
        for i in range(1, T):
            advantages[T - i - 1] = self.deltas[T - i - 1] + self.gamma * advantages[T - i]
            returns[T - i - 1] = advantages[T - i - 1] + self.values[T - i - 1]

        self.traj_memory[0].extend(self.frames)
        self.traj_memory[1].extend(self.robot_poses)
        self.traj_memory[2].extend(self.actions)
        self.traj_memory[3].extend(returns)
        self.traj_memory[4].extend(self.probs)
        self.traj_memory[5].extend(advantages)
        # self.traj_memory.append((self.states, self.actions, returns, self.probs, advantages))
        self.frames = []
        self.robot_poses = []
        self.actions = []
        self.values = []
        self.probs = []
        self.deltas = []
        self.tlen_counter = 0
        self.tcol_counter += 1

        if self.tcol_counter >= self.TRAJ_COLLECTION_NUM:
            self.update_policy()

    def get_action(self, frame, robot_pose):
        frame_in = torch.Tensor([[frame]]).to(self.train_device)
        robot_pose_in = torch.Tensor([robot_pose]).to(self.train_device)

        # if self.k % self.FRAME_SKIP == 0: # compute action every FRAME_SKIP-th frame
        value, pol = self.policy(frame_in, robot_pose_in)
        action = torch.distributions.Categorical(pol).sample()
        # else:
        #    action = self.last_action

        # self.k += 1

        # self.values.append(value.narrow(1, int(action), 1))
        # self.best_values.append(torch.max(value).detach())

        if self.train_agent:
            # store transition
            if self.last_frame is not None:
                self.frames.append(self.last_frame)
                self.robot_poses.append(self.last_robot_pose)
                self.actions.append(self.last_action)
                self.values.append(self.last_value.detach())
                self.probs.append(self.last_probs.detach())
                self.deltas.append(self.last_reward + self.gamma * value.detach() - self.last_value.detach())
                self.tlen_counter += 1
                if (self.tlen_counter >= self.TRAJ_LEN):
                    self.store_trajectory_to_memory()

            self.last_frame = frame_in
            self.last_robot_pose = robot_pose_in
            self.last_action = action
            self.last_probs = pol
            self.last_value = value

        return action

    def store_reward(self, reward, final_state):
        if not self.train_agent:  # doesn't have to save reward if not training
            return

        self.last_reward = reward

        if final_state:
            self.frames.append(self.last_frame)
            self.robot_poses.append(self.last_robot_pose)
            self.actions.append(self.last_action)
            self.values.append(self.last_value.detach())
            self.probs.append(self.last_probs.detach())
            self.deltas.append(self.last_reward - self.last_value.detach())

            self.store_trajectory_to_memory()

        # self.rewards.append(torch.Tensor([reward]))

    def update_policy(self):
        for i in range(self.EPOCH_NUM):
            # samples = self.traj_memory[i]
            # samples = random.sample(self.traj_memory, self.BATCH_SIZE)
            mb_frames, mb_robot_poses, mb_actions, mb_returns, mb_probs, mb_advantages = self.traj_memory

            new_vals, new_probs = self.policy(torch.cat(mb_frames, dim=0).to(self.train_device),
                                              torch.cat(mb_robot_poses, dim=0).to(self.train_device))
            old_probs = torch.cat(mb_probs, dim=0)

            new_pol = torch.distributions.Categorical(new_probs)
            old_pol = torch.distributions.Categorical(old_probs)

            action_tensor = torch.cat(mb_actions, dim=0)

            # new_pol2 = torch.gather(new_probs, dim=1, index=action_tensor.unsqueeze(1))
            # old_pol2 = torch.gather(old_probs, dim=1, index=action_tensor.unsqueeze(1))

            ratio = torch.exp(new_pol.log_prob(action_tensor) - old_pol.log_prob(action_tensor))
            # ratio2 = new_pol2 / old_pol2

            advantage_tensor = torch.cat(mb_advantages, dim=0)
            # advantage_tensor -= torch.mean(advantage_tensor)
            # advantage_tensor /= torch.std(advantage_tensor)

            loss_clip = -torch.min(ratio * advantage_tensor,
                                   torch.clamp(ratio, 1 - self.EPSILON, 1 + self.EPSILON) * advantage_tensor).mean()
            # loss_clip2 = torch.min(ratio2 * advantage_tensor, torch.clamp(ratio2, 1-self.EPSILON, 1+self.EPSILON) * advantage_tensor)

            returns_tensor = torch.cat(mb_returns, dim=0)

            loss_val = F.mse_loss(new_vals.squeeze(), returns_tensor)

            # loss_ent = -F.nll_loss(new_probs, action_tensor) #F.cross_entropy(new_probs, action_tensor)
            loss_ent = -new_pol.entropy().mean()

            c1, c2 = 1, 0.01

            loss = loss_clip + c1 * loss_val + c2 * loss_ent

            # loss.backward(retain_graph=True)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.traj_memory = [[], [], [], [], [], []]
        self.tcol_counter = 0
        # self.old_policy = copy.deepcopy(self.policy)

    def get_name(self):
        return self.name
