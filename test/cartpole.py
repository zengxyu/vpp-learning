# Solution of Open AI gym environment "Cartpole-v0" (https://gym.openai.com/envs/CartPole-v0) using DQN and Pytorch.
# It is is slightly modified version of Pytorch DQN tutorial from
# http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html.
# The main difference is that it does not take rendered screen as input but it simply uses observation values from the \
# environment.

import gym
import numpy as np
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

# import matplotlib.pyplot as plt

# hyper parameters
EPISODES = 1000  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.8  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 256  # NN hidden layer size
BATCH_SIZE = 64  # Q-learning batch size

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) == self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)
        # if len(self.memory) > self.capacity:
        #     del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


env = gym.make('CartPole-v0')
# env = wrappers.Monitor(env, './tmp/cartpole-v0-1')

model = Network()
if use_cuda:
    model.cuda()
memory = ReplayMemory(10000)
optimizer = optim.Adam(model.parameters(), LR)
steps_done = 0
episode_durations = []


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    # print("eps_threshold:{}".format(eps_threshold))
    steps_done += 1
    if sample > eps_threshold:
        return model(FloatTensor(state)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])


def run_episode(environment):
    for e in range(EPISODES):
        state = environment.reset()
        steps = 0
        rewards = []
        while True:
            environment.render()
            action = select_action(FloatTensor([state]))
            print(action)
            action_np = action.detach().cpu().numpy()
            next_state, reward, done, _ = environment.step(action_np[0, 0])
            # negative reward when attempt ends
            if done:
                reward = -1
            rewards.append(reward)
            memory.push((FloatTensor([state]),
                         action,  # action is already a tensor
                         FloatTensor([next_state]),
                         FloatTensor([reward])))

            learn()

            state = next_state
            steps += 1

            if done:
                # if e % 10 == 0:
                print("{2} Episode {0} finished after {1} steps"
                      .format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                print("total_reward:{}\n".format(np.sum(rewards)))
                episode_durations.append(steps)
                # plot_durations()
                break


def learn():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = torch.cat(batch_state)
    batch_action = torch.cat(batch_action)
    batch_reward = torch.cat(batch_reward).unsqueeze(1)
    batch_next_state = torch.cat(batch_next_state)
    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0].view(BATCH_SIZE, -1)
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # batch_reward + GAMMA * torch.max(model(batch_next_state).detach(), dim=1)[0].view(BATCH_SIZE, -1)
    # loss is measured from error between current and newly expected Q values
    loss_fn = nn.MSELoss()
    loss = loss_fn(current_q_values, expected_q_values)
    # loss = F.mse_loss(current_q_values, expected_q_values)
    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# def plot_durations():
#     plt.figure(2)
#     plt.clf()
#     durations_t = torch.FloatTensor(episode_durations)
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())
#
#     plt.pause(0.001)  # pause a bit so that plots are updated


# for e in range(EPISODES):
run_episode(env)

print('Complete')
env.render(close=True)
env.close()
# plt.ioff()
# plt.show()
