import os
import pickle

import torch
import torch.nn.functional as F
import numpy as np

from memory.RewardRecord import RewardRecord
from memory.ObservedMapRecord import ObservedMapRecord
from memory.RobotPoseRecord import RobotPoseRecord
from network.network_ppo_3d_unknown_map import PPOPolicy3DUnknownMap


class Agent:
    """description of class"""

    def __init__(self, params, field, summary_writer, train_agent=False, normalize=True, model_path=""):
        self.name = "PPOng"
        self.train_agent = train_agent
        self.normalize = normalize

        self.FRAME_SIZE = np.product(field.shape)
        self.ACTION_SPACE = len(params['action'])
        self.TRAJ_COLLECTION_NUM = params['traj_collection_num']
        self.TRAJ_LEN = params['traj_len']
        self.tlen_counter = 0
        self.tcol_counter = 0
        self.EPSILON = 0.2
        self.EPOCH_NUM = 4
        self.gamma = params['gamma']
        self.train_device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
            'cpu')  # 'cuda' if torch.cuda.is_available() else pu'
        self.model = params['model']
        self.policy = self.model(self.ACTION_SPACE).to(self.train_device)
        print(self.policy)
        if model_path != "":
            self.policy.load_state_dict(torch.load(model_path, self.train_device))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=params['lr'])

        # lam = lambda f: 1 - f / train_steps
        # self.opti_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=300, gamma=0.05, last_epoch=-1)
        self.summary_writer = summary_writer
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

        if self.normalize:
            # self.reward_record = RewardRecord()
            # self.observed_map_record = ObservedMapRecord()
            # self.robot_pose_record = RobotPoseRecord()
            self.observed_map_mean, self.observed_map_std, self.robot_pose_mean, self.robot_pose_std, self.reward_mean, self.reward_std = self.load_data_mean_std(
                config_dir=params['config_dir'])

        print("\nis cuda available:", torch.cuda.is_available())
        print("train device:", self.train_device)
        print("TRAJ_COLLECTION_NUM:", self.TRAJ_COLLECTION_NUM)
        print("TRAJ_LEN:", self.TRAJ_LEN)
        print("gamma:", self.gamma)
        print("lr:", params['lr'])

    def load_data_mean_std(self, config_dir):
        with open(os.path.join(config_dir, "observed_map_mean_std.pkl"), 'rb') as f:
            observed_map_mean, observed_map_std = pickle.load(f)

        with open(os.path.join(config_dir, "robot_pose_mean_std.pkl"), 'rb') as f:
            robot_pose_mean, robot_pose_std = pickle.load(f)

        with open(os.path.join(config_dir, "reward_mean_std1.pkl"), 'rb') as f:
            reward_mean, reward_std = pickle.load(f)
        print(
            "observed_map_mean, observed_map_std, robot_pose_mean, robot_pose_std, reward_mean, reward_std:{},{},{},{},{},{}".format(
                observed_map_mean, observed_map_std, robot_pose_mean, robot_pose_std, reward_mean, reward_std))
        return observed_map_mean, observed_map_std, robot_pose_mean, robot_pose_std, reward_mean, reward_std

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
        if self.normalize:
            # self.observed_map_record.add_observed_map(frame)
            # self.robot_pose_record.add_robot_pose(robot_pose)
            #
            # observed_map_mean = self.observed_map_record.get_observed_map_mean()
            # observed_map_std = self.observed_map_record.get_observed_map_std()
            #
            # robot_pose_mean = self.robot_pose_record.get_robot_pose_mean()
            # robot_pose_std = self.robot_pose_record.get_robot_pose_std()

            frame = (frame - self.observed_map_mean) / (self.observed_map_std + 1e-8)
            robot_pose = (robot_pose - self.robot_pose_mean) / (self.robot_pose_std + 1e-8)

        frame_in = torch.Tensor([frame]).to(self.train_device)
        robot_pose_in = torch.Tensor([robot_pose]).to(self.train_device)

        value, pol = self.policy( frame_in,robot_pose_in)
        action = torch.distributions.Categorical(pol).sample()

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

        if self.normalize:
            # self.reward_record.add_reward(reward)
            # reward_std = self.reward_record.get_reward_std()
            # reward_mean = self.reward_record.get_reward_mean()
            # reward = (reward - reward_mean) / (reward_std + 1e-8)
            reward = reward / (self.reward_std + 1e-8)

        # reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        self.last_reward = reward

        if final_state:
            self.frames.append(self.last_frame)
            self.robot_poses.append(self.last_robot_pose)
            self.actions.append(self.last_action)
            self.values.append(self.last_value.detach())
            self.probs.append(self.last_probs.detach())
            self.deltas.append(self.last_reward - self.last_value.detach())

            self.store_trajectory_to_memory()

    def update_policy(self):
        loss_values = []
        loss_clip_values = []
        loss_val_values = []
        loss_ent_values = []

        # for i in range(self.EPOCH_NUM):
            # samples = self.traj_memory[i]
            # samples = random.sample(self.traj_memory, self.BATCH_SIZE)
            # mb_frames, mb_robot_poses, mb_actions, mb_returns, mb_probs, mb_advantages = self.traj_memory
            #
            # new_vals, new_probs = self.policy(torch.cat(mb_robot_poses, dim=0).to(self.train_device))
            # old_probs = torch.cat(mb_probs, dim=0)
            #
            # new_pol = torch.distributions.Categorical(new_probs)
            # old_pol = torch.distributions.Categorical(old_probs)
            #
            # action_tensor = torch.cat(mb_actions, dim=0)
            #
            # # new_pol2 = torch.gather(new_probs, dim=1, index=action_tensor.unsqueeze(1))
            # # old_pol2 = torch.gather(old_probs, dim=1, index=action_tensor.unsqueeze(1))
            #
            # ratio = torch.exp(new_pol.log_prob(action_tensor) - old_pol.log_prob(action_tensor))
            # # ratio2 = new_pol2 / old_pol2
            #
            # advantage_tensor = torch.cat(mb_advantages, dim=0)
            # # advantage_tensor -= torch.mean(advantage_tensor)
            # # advantage_tensor /= torch.std(advantage_tensor)
            #
            # loss_clip = -torch.min(ratio * advantage_tensor,
            #                        torch.clamp(ratio, 1 - self.EPSILON, 1 + self.EPSILON) * advantage_tensor).mean()
            # # loss_clip2 = torch.min(ratio2 * advantage_tensor, torch.clamp(ratio2, 1-self.EPSILON, 1+self.EPSILON) * advantage_tensor)
            #
            # returns_tensor = torch.cat(mb_returns, dim=0)
            #
            # loss_val = F.mse_loss(new_vals.squeeze(), returns_tensor)
            #
            # # loss_ent = -F.nll_loss(new_probs, action_tensor) #F.cross_entropy(new_probs, action_tensor)
            # loss_ent = -new_pol.entropy().mean()
            #
            # c1, c2 = 1, 0.01
            #
            # loss = loss_clip + c1 * loss_val + c2 * loss_ent
            #
            # # loss.backward(retain_graph=True)
            # loss.backward()
            #
            # for p in self.policy.parameters():
            #     p.grad.data.clamp_(-1, 1)
            #
            # self.optimizer.step()
            # self.optimizer.zero_grad()

        for i in range(self.EPOCH_NUM):
            mb_frames, mb_robot_poses, mb_actions, mb_returns, mb_probs, mb_advantages = self.traj_memory

            new_vals, new_probs = self.policy(torch.cat(mb_frames, dim=0).to(self.train_device),
                                              torch.cat(mb_robot_poses, dim=0).to(self.train_device))
            # new_vals, new_probs = self.policy(torch.cat(mb_robot_poses, dim=0).to(self.train_device))
            old_probs = torch.cat(mb_probs, dim=0)

            new_pol = torch.distributions.Categorical(new_probs)
            old_pol = torch.distributions.Categorical(old_probs)

            action_tensor = torch.cat(mb_actions, dim=0)

            ratio = torch.exp(new_pol.log_prob(action_tensor) - old_pol.log_prob(action_tensor))

            advantage_tensor = torch.cat(mb_advantages, dim=0)

            c1, c2 = 1, 0.01

            loss_clip = -torch.min(ratio * advantage_tensor,
                                   torch.clamp(ratio, 1 - self.EPSILON, 1 + self.EPSILON) * advantage_tensor).mean()

            returns_tensor = torch.cat(mb_returns, dim=0)

            loss_val = c1 * F.mse_loss(new_vals.squeeze(), returns_tensor)

            loss_ent = -c2 * new_pol.entropy().mean()

            loss = loss_clip + loss_val + loss_ent

            loss.backward()

            loss_values.append(loss.item())

            loss_clip_values.append(loss_clip.item())

            loss_val_values.append(loss_val.item())

            loss_ent_values.append(loss_ent.item())

            self.optimizer.step()
            self.optimizer.zero_grad()

        # self.opti_scheduler.step()

        self.traj_memory = [[], [], [], [], [], []]
        self.tcol_counter = 0

        self.summary_writer.add_loss(np.mean(loss_values))
        self.summary_writer.add_3_loss(np.mean(loss_clip_values), np.mean(loss_val_values),
                                       np.mean(loss_ent_values), np.mean(loss_values))

    def get_name(self):
        return self.name
