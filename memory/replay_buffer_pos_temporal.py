import random
from collections import deque, namedtuple
import numpy as np
import torch

from memory.data_structures import SumTree


class PriorityReplayBufferPoseTemporal:
    """
    A special replay buffer which stores a priority for each event and only overwrites the ones with lowest priority

    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py

    Taken from: https://github.com/txzhao/rl-zoo/blob/master/DQN/priorExpReplay.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, buffer_size, batch_size, device, is_discrete, seed=1337):
        self.tree = SumTree(buffer_size)
        self.buffer_size = buffer_size
        self.size = 0
        self.batch_size = batch_size
        self.device = device
        self.is_discrete = is_discrete
        self.seed = random.seed(seed)

    def add_experience(self, state, action, reward, next_state, done):
        transition = np.array(
            [state[0], state[1], action, reward, next_state[0], next_state[1], done],
            dtype=np.object)
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0.0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p
        self.size += 1

    def preload_experiences(self, experience):
        vs, transitions = experience
        for v, transition in zip(vs, transitions):
            self.tree.add(v, transition)
            self.size += 1

    def get_all_experiences(self):
        vs = self.tree.tree[-self.tree.capacity:]
        transitions = self.tree.data[-self.tree.capacity:]
        return vs[:self.size], transitions[:self.size]

    def transform(self, data_slice):
        robot_poses_in = []
        next_robot_poses_in = []
        for i in range(data_slice.shape[0]):
            frame, pose, action, reward, next_frame, next_pose, done = data_slice[i]
            robot_poses_in.append(pose)
            next_robot_poses_in.append(next_pose)
        cur_frame, cur_pose, cur_action, cur_reward, cur_next_frame, cur_next_pose, cur_done = data_slice[-1]

        frames_in = cur_frame
        robot_poses_in = np.array(robot_poses_in, dtype=np.float)
        action = cur_action
        reward = cur_reward
        next_frames_in = cur_next_frame
        next_robot_poses_in = np.array(next_robot_poses_in, dtype=np.float)
        done = cur_done

        return np.array([frames_in, robot_poses_in, action, reward, next_frames_in, next_robot_poses_in, done],
                        dtype=np.object)

    def sample(self, is_vpp=True, num_experiences=None, is_weighted=True):
        # Draws a random sample of experience from the replay buffer
        batch_size = self.batch_size if num_experiences is None else num_experiences
        b_idx, b_memory, ISWeights = np.empty((batch_size,), dtype=np.int32), np.empty(
            (batch_size, self.tree.data[0].size), dtype=object), np.empty((batch_size, 1), dtype=np.float)
        pri_seg = self.tree.total_p / float(batch_size)  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        if self.size > self.tree.capacity:
            min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        else:
            min_prob = np.min(self.tree.tree[
                              self.tree.capacity - 1:self.tree.capacity - 1 + self.size]) / self.tree.total_p  # for later calculate ISweight
        if is_weighted:
            for i in range(batch_size):
                a, b = pri_seg * i, pri_seg * (i + 1)
                v = random.uniform(a, b)
                idx, p, data_idx, data = self.tree.get_leaf(v)
                prob = p / self.tree.total_p
                ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
                data_idx = max(4, data_idx)
                transformed_data = self.transform(self.tree.data[data_idx - 4:data_idx + 1])
                b_idx[i], b_memory[i, :] = idx, transformed_data

        else:
            for i in range(batch_size):
                idx, p, data = self.tree.get_leaf_random()
                prob = p / self.tree.total_p
                ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
                b_idx[i], b_memory[i, :] = idx, data
        b_memory = self.to_tensor(b_memory, is_vpp)
        return torch.from_numpy(b_idx).to(self.device), b_memory, torch.from_numpy(ISWeights).to(self.device)

    def to_tensor(self, mini_batch, is_vpp):
        mini_batch = mini_batch.T
        frames_in, robot_poses_in, actions, rewards, next_frames_in, next_robot_poses_in, dones = mini_batch

        frames_in = self.to_float_tensor(frames_in)
        robot_poses_in = self.to_float_tensor(robot_poses_in)

        next_frames_in = self.to_float_tensor(next_frames_in)
        next_robot_poses_in = self.to_float_tensor(next_robot_poses_in)

        state_in = [frames_in, robot_poses_in]
        next_state_in = [next_frames_in, next_robot_poses_in]

        actions = self.to_float_tensor(actions)
        rewards = self.to_float_tensor(rewards)
        dones = self.to_long_tensor(dones)
        if self.is_discrete:
            actions = actions.long()
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)
        if len(rewards.shape) == 1:
            rewards = rewards.unsqueeze(1)
        if len(dones.shape) == 1:
            dones = dones.unsqueeze(1)
        return [state_in, actions, rewards, next_state_in, dones]

    def to_float_tensor(self, object_array):
        return torch.FloatTensor(np.array(object_array.tolist()).astype(np.float)).to(self.device)

    def to_long_tensor(self, object_array):
        return torch.LongTensor(np.array(object_array.tolist()).astype(np.long)).to(self.device)

    def batch_update(self, tree_idx, abs_errors):
        # Update the priority of the given index
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return min(self.size, self.buffer_size)