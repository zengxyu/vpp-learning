import os
import time
import math
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import agents.agent as agent
import torch.nn.functional as F
import utils.replay_buffers as mem
#import torchvision.transforms as T


class NN(nn.Module):
	# A simple neural network

	def __init__(self, input_size, hidden1_size, hidden2_size, output_size, seed):
		super(NN, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.seed = torch.manual_seed(seed)
		#
		self.fc_in = nn.Linear(input_size, hidden1_size)		# input layer -> hidden layer 1 (256)
		self.fc_h1 = nn.Linear(hidden1_size, hidden2_size)		# hidden layer 1 (256) -> hidden layer 2 (128)
		self.fc_out = nn.Linear(hidden2_size, output_size)		# hidden layer 2 (128) -> output layer

	def forward(self, x):
		# Forwards a state through the NN to get an action

		x = self.fc_in(x)  # input layer -> hidden layer 1
		x = F.relu(x)

		x = self.fc_h1(x)  # hidden layer 1 -> hidden layer 2
		x = torch.tanh(x)  # x = F.relu(x)

		x = self.fc_out(x)	# hidden layer 2 -> output layer
		return x


class ConvolutionNN(nn.Module):
	# A simple convolutional neural network

	def __init__(self, input_size, _, __, output_size, seed):
		super(ConvolutionNN, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.seed = torch.manual_seed(seed)
		
		self.features = nn.Sequential(
			nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.ReLU()
		)
		
		size_x = input_size[1]
		size_y = input_size[2]
		
		last_layer = None
		for layer in self.features:
			if type(layer) is nn.Conv2d:
				size_x = ((size_x - layer.kernel_size[0]) // layer.stride[0]) + 1
				size_y = ((size_y - layer.kernel_size[1]) // layer.stride[1]) + 1
				last_layer = layer
		
		self.fc = nn.Sequential(
			nn.Linear(size_x*size_y*last_layer.out_channels, 256),
			nn.ReLU(),
			nn.Linear(256, output_size)
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


class DuelingNN(nn.Module):
	# A simple dueling neural network

	def __init__(self, input_size, hidden1_size, hidden2_size, output_size, seed):
		super(DuelingNN, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.seed = torch.manual_seed(seed)
		#
		self.fc_in = nn.Linear(input_size, hidden1_size)
		self.fc_h1 = nn.Linear(hidden1_size, hidden2_size)
		self.fc_V = nn.Linear(hidden2_size, 1)
		self.fc_A = nn.Linear(hidden2_size, output_size)

	def forward(self, x):
		# Forwards a state through the NN to get an action

		flat1 = F.relu(self.fc_in(x))
		flat2 = F.relu(self.fc_h1(flat1))
		V = self.fc_V(flat2)
		A = self.fc_A(flat2)

		return V, A


class Agent(agent.IAgent):
	# A Deep-Q Network
	#	Double DQN: Use 2 networks to decouple the action selection from the target Q value generation (Reduces the overestimation of q values, more stable learning)
	#		Implemented in DQN.update
	#	Dueling DQN: Only learn when actions have an effect on the rewards (Good if there is a gamestate when any input is accepted)
	#		Implemented in NN.forward
	#		TODO: https://nervanasystems.github.io/coach/components/agents/value_optimization/dueling_dqn.html
	#	PER Prioritized Experience Replay: Very important memories may occur rarely thus needing prioritization
	#		Implemented in ReplayBuffer

	NAME = "DQN"

	# Hyperparameters
	LAYER_H1_SIZE = 256
	LAYER_H2_SIZE = 128
	ALPHA = 0.00005			# learning rate (0.001, 0.0001)
	ALPHA_DECAY = 0.01		# [UNUSED] for Adam
	GAMMA = 0.99			# discount factor (0.95)
	EPSILON_MAX = 1.0		# epsilon greedy threshold
	EPSILON_MIN = 0.02
	EPSILON_DECAY = 30000	# amount of steps to reach half-life (0.99 ~~ 400 steps)
	TAU = 0.001				# target update factor for double DQN (0.002)
	TGT_UPDATE_RATE = 1000	# target update rate ^ instead of factor
	MEMORY_SIZE = 100000	# size of the replay buffer
	MEMORY_FILL = 10000		# size of samples to get before starting to play
	BATCH_SIZE = 64			# size of one mini-batch to sample (128)
	UPDATE_RATE = 2			# update every X steps (1)
	DOUBLE_DQN = True		# whether to use double-DQN or vanilla DQN
	DUELING_DQN = False		# WIP: whether to use dueling NN or normal NN
	PRIO_REPLAY = False		# whether to use prioritized replay buffer or normal replay buffer

	"""
	LAYER_H1_SIZE = 64			
	LAYER_H2_SIZE = 64		
	ALPHA = 0.0005			# learning rate (0.001)
	ALPHA_DECAY = 0.01		# [UNUSED] for Adam
	GAMMA = 0.99			# discount factor (0.95)
	EPSILON_MAX = 1.0		# epsilon greedy threshold
	EPSILON_MIN = 0.01
	EPSILON_DECAY = 0.995	# (0.995)
	TAU = 0.001				# target update factor for double DQN (0.002)
	MEMORY_SIZE = 200000	# size of the replay 
	BATCH_SIZE = 64			# size of one mini-batch to sample (128)
	UPDATE_RATE = 4			# update every X steps (1)
	DOUBLE_DQN = True		# whether to use double-DQN or vanilla DQN
	PRIO_REPLAY = False		# whether to use prioritized replay buffer or normal replay buffer
	"""

	def __init__(self, input_size, output_size, training_mode, is_conv, load_filename, seed):
		self.update_step = 0
		self.epsilon = self.EPSILON_MAX
		self.seed = seed
		self.training_mode = training_mode
		self.is_conv = is_conv
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.input_size = input_size
		self.output_size = output_size
		self.load_filename = load_filename

		# Handle loading of previously saved models
		if load_filename:
			with open(load_filename, "rb") as f:
				self.epsilon = pickle.load(f)
				#self.memory = pickle.load(f)
				self.LAYER_H1_SIZE = pickle.load(f)
				self.LAYER_H2_SIZE = pickle.load(f)
				self.ALPHA = pickle.load(f)
				self.ALPHA_DECAY = pickle.load(f)
				self.GAMMA = pickle.load(f)
				self.EPSILON_MAX = pickle.load(f)
				self.EPSILON_MIN = pickle.load(f)
				self.EPSILON_DECAY = pickle.load(f)
				self.TAU = pickle.load(f)
				self.MEMORY_SIZE = pickle.load(f)
				self.BATCH_SIZE = pickle.load(f)
				self.UPDATE_RATE = pickle.load(f)
				self.DOUBLE_DQN = pickle.load(f)
				self.PRIO_REPLAY = pickle.load(f)
				# self.MEMORY_FILL = self.MEMORY_SIZE		# if a model is loaded we re-fill our experience buffer completely before learning

		# Start epsilon at EPSILON_MIN when loading an existing model and model is updated
		self.step = math.inf if self.load_filename else 0

		if self.PRIO_REPLAY:
			self.memory = mem.PriorityReplayBuffer(self.device, self.input_size, self.MEMORY_SIZE, self.BATCH_SIZE, self.seed)
		else:
			self.memory = mem.ReplayBuffer(self.device, self.MEMORY_SIZE, self.BATCH_SIZE, self.seed)

	def init_network(self):
		# Create the model / NN
		ModelType = NN
		if self.DUELING_DQN:
			ModelType = DuelingNN
		elif self.is_conv:
			ModelType = ConvolutionNN

		self.nn = ModelType(self.input_size, self.LAYER_H1_SIZE, self.LAYER_H2_SIZE, self.output_size, self.seed).to(self.device)
		self.target_nn = ModelType(self.input_size, self.LAYER_H1_SIZE, self.LAYER_H2_SIZE, self.output_size, self.seed).to(self.device)

		# After creating all structures, load the weights into the NN's
		if self.load_filename:
			self.nn.load_state_dict(torch.load(self.load_filename + ".nn.pth", map_location=self.device))
			if self.DOUBLE_DQN:
				self.target_nn.load_state_dict(torch.load(self.load_filename + ".target_nn.pth", map_location=self.device))

		self.optimizer = optim.Adam(self.nn.parameters(), lr=self.ALPHA, amsgrad=False)
		self.loss_func = nn.MSELoss()
		self.td_loss_func = nn.L1Loss(reduction='none')

	def save_model(self, filename):
		torch.save(self.nn.state_dict(), filename + ".mdl.nn.pth")
		if self.DOUBLE_DQN:
			torch.save(self.target_nn.state_dict(), filename + ".mdl.target_nn.pth")
		with open(filename + ".mdl", "wb") as f:
			pickle.dump(self.epsilon, f)
			#pickle.dump(self.memory, f)
			pickle.dump(self.LAYER_H1_SIZE, f)
			pickle.dump(self.LAYER_H2_SIZE, f)
			pickle.dump(self.ALPHA, f)
			pickle.dump(self.ALPHA_DECAY, f)
			pickle.dump(self.GAMMA, f)
			pickle.dump(self.EPSILON_MAX, f)
			pickle.dump(self.EPSILON_MIN, f)
			pickle.dump(self.EPSILON_DECAY, f)
			pickle.dump(self.TAU, f)
			pickle.dump(self.MEMORY_SIZE, f)
			pickle.dump(self.BATCH_SIZE, f)
			pickle.dump(self.UPDATE_RATE, f)
			pickle.dump(self.DOUBLE_DQN, f)
			pickle.dump(self.PRIO_REPLAY, f)

	def copy_from(self, model):
		self.nn.load_state_dict(model.nn.state_dict())
		if self.DOUBLE_DQN:
			self.target_nn.load_state_dict(model.target_nn.state_dict())

	def get_action(self, state):
		# Extract an output tensor (action) by forwarding the state into the NN

		# Explore (random action)
		if self.training_mode and random.random() < self.epsilon:
			return random.choice(np.arange(self.nn.output_size))

		# Abuse (get action from NN)
		state = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)

		if self.DUELING_DQN:
			self.nn.eval()						# Set NN to eval mode
			with torch.no_grad():				# Disable autograd engine
				_, advantage = self.nn(state)	# Forward state to NN
			self.nn.train()						# Set NN back to train mode

			return torch.argmax(advantage).item()
		else:
			self.nn.eval()				# Set NN to eval mode
			with torch.no_grad():		# Disable autograd engine
				value = self.nn(state)	# Forward state to NN
			self.nn.train()				# Set NN back to train mode

			# Return the action with the highest tensor value
			action_max_value, index = torch.max(value, 1)

			return index.item()

	def update(self, state, action, reward, next_state, done):
		# Stores the experience in the replay buffer
		self.memory.add_experience(state, action, reward, next_state, done)

		# Only update every 'UPDATE_RATE' steps if episode is not over to train faster
		self.update_step += 1
		if (self.update_step) % self.UPDATE_RATE != 0 and not done:
			return

		# Wait for the replay memory to be filled with a few experiences first before learning from it
		if len(self.memory) < self.MEMORY_FILL:
			return

		# Extract a random batch of experiences
		if self.PRIO_REPLAY:
			tree_idx, batch, IS_weights = self.memory.sample()
			states, actions, rewards, next_states, dones = batch
		else:
			states, actions, rewards, next_states, dones = self.memory.sample()

		# Normalize rewards / gradient step size
		#rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		if self.DUELING_DQN:
			indices = np.arange(self.BATCH_SIZE)

			curr_V, curr_A = self.nn(states)
			next_V, next_A = self.target_nn(next_states)
			eval_V, eval_A = self.nn(next_states)

			# Calculate current Q-Value from current states (current prediction of the NN)
			curr_Q = torch.add(curr_V, (curr_A - curr_A.mean(dim=1, keepdim=True)))[indices, actions]
			q_next = torch.add(next_V, (next_A - next_A.mean(dim=1, keepdim=True)))
			q_eval = torch.add(eval_V, (eval_A - eval_A.mean(dim=1, keepdim=True)))

			max_actions = torch.argmax(q_eval, dim=1)

			# Calculate Q-Target values
			expected_Q = (1 - dones) * (q_next[indices, max_actions] * self.GAMMA) + rewards
		else:
			# Calculate current Q-Value from current states (current prediction of the NN)
			curr_Q = self.nn(states).gather(1, actions)

			if self.DOUBLE_DQN:
				# Double DQN
				#next_Q = self.target_nn(next_states).detach()
				#max_next_Q = next_Q.max(1)[0].unsqueeze(1)

				next_q_values = self.nn(next_states)
				next_q_state_values = self.target_nn(next_states)
				next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1))

				expected_Q = (1 - dones) * (next_q_value * self.GAMMA) + rewards
			else:
				# Vanilla DQN
				next_Q = self.nn(next_states).detach()
				max_next_Q = next_Q.max(1)[0].unsqueeze(1)

				# Calculate Q-Target values
				expected_Q = (1 - dones) * (max_next_Q * self.GAMMA) + rewards

		# Calculate the loss and update priorities in replay buffer if needed
		if self.PRIO_REPLAY:
			td_errors = self.td_loss_func(curr_Q, expected_Q)
			loss = (IS_weights * self.loss_func(curr_Q, expected_Q)).mean()
			self.memory.batch_update(tree_idx, td_errors.cpu().detach().numpy())
		else:
			loss = self.loss_func(curr_Q, expected_Q.detach())

		# Minimize the loss
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# Update exploration rate
		#self.epsilon = max(self.EPSILON_MIN, self.epsilon*EPSILON_DECAY_FAC)
		self.epsilon = self.EPSILON_MIN + (self.EPSILON_MAX - self.EPSILON_MIN) * math.exp(-1.0 * self.step / self.EPSILON_DECAY)
		self.step += 1

		if self.DOUBLE_DQN:
			# Update target NN from NN each 'TGT_UPDATE_RATE' steps
			if self.update_step % self.TGT_UPDATE_RATE == 0:
				self.target_nn.load_state_dict(self.nn.state_dict())
			#
			# Slowly update target NN from NN using factor TAU
			# for target_param, param in zip(self.target_nn.parameters(), self.nn.parameters()):
			#	target_param.data.copy_(self.TAU * param + (1 - self.TAU) * target_param)
