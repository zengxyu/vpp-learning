import random
import torch
import numpy as np
from collections import namedtuple, deque
from utils.data_structures import SumTree

class ReplayBuffer:
	# Replay buffer to store past experiences that the agent can then use for training data => de-correlation
	
	def __init__(self, device, buffer_size=200000, batch_size=32, seed=1337):
		self.device = device
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
		self.seed = random.seed(seed)
	
	def add_experience(self, states, actions, rewards, next_states, dones):
		# Adds experience(s) into the replay buffer
		if type(dones) == list:
			assert type(dones[0]) != list, "A done shouldn't be a list"
			experiences = [self.experience(state, action, reward, next_state, done)
							for state, action, reward, next_state, done in
							zip(states, actions, rewards, next_states, dones)]
			self.memory.extend(experiences)
		else:
			experience = self.experience(states, actions, rewards, next_states, dones)
			self.memory.append(experience)
	
	def sample(self, num_experiences=None, separate_out_data_types=True):
		# Draws a random sample of experience from the replay buffer
		experiences = self.pick_experiences(num_experiences)
		if separate_out_data_types:
			states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
			return states, actions, rewards, next_states, dones
		else:
			return experiences
	
	def separate_out_data_types(self, experiences):
		# Puts the sampled experience into the correct format for a PyTorch neural network
		
		states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
		next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
		dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)
		
		return states, actions, rewards, next_states, dones
	
	def pick_experiences(self, num_experiences=None):
		# Extract a fixed amount of experiences randomly from the memory
		batch_size = self.batch_size if num_experiences is None else num_experiences
		return random.sample(self.memory, k=batch_size)
	
	def __len__(self):
		return len(self.memory)

class PriorityReplayBuffer:
	"""
	A special replay buffer which stores a priority for each event and only overwrites the ones with lowest priority
	
	This SumTree code is modified version and the original code is from:
	https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
	
	Taken from: https://github.com/txzhao/rl-zoo/blob/master/DQN/priorExpReplay.py
	"""
	
	epsilon = 0.01		# small amount to avoid zero priority
	alpha = 0.6			# [0~1] convert the importance of TD error to priority
	beta = 0.4			# importance-sampling, from initial value increasing to 1
	beta_increment_per_sampling = 0.001
	abs_err_upper = 1.0	# clipped abs error
	size = 0
	
	def __init__(self, device, input_size, buffer_size=200000, batch_size=32, seed=1337):
		self.device = device
		self.input_size = input_size
		self.tree = SumTree(buffer_size)
		self.batch_size = batch_size
		self.seed = random.seed(seed)
	
	def add_experience(self, s, a, r, s_, d):
		# Adds experience(s) into the replay buffer
		transition = np.hstack((s, [a, r, d], s_))
		max_p = np.max(self.tree.tree[-self.tree.capacity:])
		if max_p == 0.0:
			max_p = self.abs_err_upper
		self.tree.add(max_p, transition)   # set the max p for new p
		self.size += 1
	
	def sample(self, num_experiences=None):
		# Draws a random sample of experience from the replay buffer
		batch_size = self.batch_size if num_experiences is None else num_experiences
		b_idx, b_memory, ISWeights = np.empty((batch_size,), dtype=np.int32), np.empty((batch_size, self.tree.data[0].size)), np.empty((batch_size, 1))
		pri_seg = self.tree.total_p / float(batch_size)       # priority segment
		self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
		
		if self.size > self.tree.capacity:
			min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
		else:
			min_prob = np.min(self.tree.tree[self.tree.capacity-1:self.tree.capacity-1+self.size]) / self.tree.total_p     # for later calculate ISweight
		
		for i in range(batch_size):
			a, b = pri_seg * i, pri_seg * (i + 1)
			v = random.uniform(a, b)
			idx, p, data = self.tree.get_leaf(v)
			prob = p / self.tree.total_p
			ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
			b_idx[i], b_memory[i, :] = idx, data
		
		# b_idx, (states, actions, rewards, next_states, dones), weights
		return b_idx, \
				(torch.FloatTensor(b_memory[:, :self.input_size]).to(self.device), \
				torch.LongTensor(b_memory[:, self.input_size:self.input_size+1].astype(int)).to(self.device), \
				torch.FloatTensor(b_memory[:, self.input_size+1:self.input_size+2]).to(self.device), \
				torch.FloatTensor(b_memory[:, -self.input_size:]).to(self.device), \
				torch.FloatTensor(b_memory[:, self.input_size+2:self.input_size+3]).to(self.device)), \
			torch.FloatTensor(ISWeights).to(self.device)
	
	def batch_update(self, tree_idx, abs_errors):
		# Update the priority of the given index
		abs_errors += self.epsilon  # convert to abs and avoid 0
		clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
		ps = np.power(clipped_errors, self.alpha)
		for ti, p in zip(tree_idx, ps):
			self.tree.update(ti, p)
	
	def __len__(self):
		return min(self.size, self.buffer_size)
