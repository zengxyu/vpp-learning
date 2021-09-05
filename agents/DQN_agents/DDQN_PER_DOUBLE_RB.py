import pickle
from random import random

import torch
import torch.nn.functional as F
from torch import optim

from agents.Base_Agent import Base_Agent
from agents.Base_Agent_DQN import Base_Agent_DQN
from agents.DQN_agents.DDQN import DDQN
import memory
import numpy as np
import os

from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.basic_logger import BasicLogger


class DDQN_PER_DOUBLE_RB(Base_Agent):
    """A DQN agent with prioritised experience replay"""
    agent_name = "DDQN with Prioritised Replay"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.hyper_parameters = config.hyper_parameters['DQN_Agents']
        self.memory = memory.replay_buffer.DoublePriorityReplayBuffer(buffer_size=self.hyper_parameters['buffer_size'],
                                                                      batch_size=self.hyper_parameters['batch_size'],
                                                                      device=self.device, is_discrete=True,
                                                                      seed=self.seed)
        self.q_network_target_unknown = self.create_NN_unknown()
        self.q_network_local_unknown = self.create_NN_unknown()
        Base_Agent_DQN.copy_model_over(from_model=self.q_network_target_unknown, to_model=self.q_network_target_unknown)
        self.q_network_optimizer_unknown = optim.Adam(self.q_network_target_unknown.parameters(),
                                                      lr=self.hyper_parameters["learning_rate"], eps=1e-4)

        self.q_network_target_known = self.create_NN_known()
        self.q_network_local_known = self.create_NN_known()
        Base_Agent_DQN.copy_model_over(from_model=self.q_network_target_known, to_model=self.q_network_target_known)
        self.q_network_optimizer_known = optim.Adam(self.q_network_target_known.parameters(),
                                                    lr=self.hyper_parameters["learning_rate"], eps=1e-4)

        self.exploration_strategy = Epsilon_Greedy_Exploration(self.hyper_parameters)

        self.model_dict, self.optimizer_dict = self.__build_model_and_optimizer_dict()

    def create_NN_unknown(self):
        Network = self.config.network[0]
        net = Network(action_size=self.config.environment['action_size']).to(self.device)
        return net

    def create_NN_known(self):
        Network = self.config.network[1]
        net = Network(action_size=self.config.environment['action_size']).to(self.device)
        return net

    def step(self, state, action, reward, next_state, done):
        self.memory.add_experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.global_step_number += 1

    def learn(self):
        loss_unknown = self.learn_unknown()
        loss_known = self.learn_known()
        return loss_unknown, loss_known

    def learn_unknown(self):
        """Runs a learning iteration for the Q network after sampling from the replay buffer in a prioritised way"""
        # sampled_experiences, importance_sampling_weights = self.memory.sample()
        tree_idx, minibatch, ISWeights = self.memory.sample_from_unknown(is_vpp=self.config.environment['is_vpp'])
        states, actions, rewards, next_states, dones = minibatch
        rewards = rewards[0]
        loss, td_errors = self.compute_loss_and_td_errors(self.q_network_local_unknown, self.q_network_target_unknown,
                                                          states, next_states,
                                                          rewards, actions, dones, ISWeights)
        self.take_optimisation_step(self.q_network_optimizer_unknown, self.q_network_local_unknown, loss,
                                    self.hyper_parameters["gradient_clipping_norm"])

        self.update_memory_batch_errors(tree_idx, td_errors, rewards, is_unknown=True)
        # self.soft_update_of_target_network(self.q_network_local, self.q_network_target, self.hyperparameters["tau"])
        self.skipping_step_update_of_target_network(self.q_network_local_unknown, self.q_network_target_unknown,
                                                    global_step_number=self.global_step_number,
                                                    update_every_n_steps=self.hyper_parameters["update_every_n_steps"])
        if (self.global_step_number + 1) % 10000 == 0:
            pickle.dump(self.memory, open(os.path.join(self.config.folder['exp_sv'], "buffer.obj"), 'wb'))
            print("save replay buffer to local")
        # print(loss.detach().cpu().numpy(), torch.mean(torch.abs(td_errors)).detach().cpu().numpy())

        return loss.detach().cpu().numpy()

    def learn_known(self):
        """Runs a learning iteration for the Q network after sampling from the replay buffer in a prioritised way"""
        tree_idx, minibatch, ISWeights = self.memory.sample_from_known(is_vpp=self.config.environment['is_vpp'])
        states, actions, rewards, next_states, dones = minibatch
        rewards = rewards[1]
        loss, td_errors = self.compute_loss_and_td_errors(self.q_network_local_known, self.q_network_target_known,
                                                          states, next_states,
                                                          rewards, actions, dones, ISWeights)
        self.take_optimisation_step(self.q_network_optimizer_known, self.q_network_local_known, loss,
                                    self.hyper_parameters["gradient_clipping_norm"])

        self.update_memory_batch_errors(tree_idx, td_errors, rewards, is_unknown=False)
        # self.soft_update_of_target_network(self.q_network_local, self.q_network_target, self.hyperparameters["tau"])
        self.skipping_step_update_of_target_network(self.q_network_local_known, self.q_network_target_known,
                                                    global_step_number=self.global_step_number,
                                                    update_every_n_steps=self.hyper_parameters["update_every_n_steps"])
        if (self.global_step_number + 1) % 10000 == 0:
            pickle.dump(self.memory, open(os.path.join(self.config.folder['exp_sv'], "buffer.obj"), 'wb'))
            print("save replay buffer to local")
        # print(loss.detach().cpu().numpy(), torch.mean(torch.abs(td_errors)).detach().cpu().numpy())

        return loss.detach().cpu().numpy()

    def pick_action_unknown(self, state):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        state_tensor = []
        for s in state:
            s_tensor = torch.Tensor([s]).to(self.device)
            state_tensor.append(s_tensor)
        state = state_tensor

        self.q_network_local_unknown.eval()
        with torch.no_grad():
            action_values = self.q_network_local_unknown(state)
        self.q_network_local_unknown.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})

        return action

    def pick_action_known(self, state):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        state_tensor = []
        for s in state:
            s_tensor = torch.Tensor([s]).to(self.device)
            state_tensor.append(s_tensor)
        state = state_tensor

        self.q_network_local_known.eval()
        with torch.no_grad():
            action_values = self.q_network_local_known(state)
            # a = action_values.detach().cpu().numpy()
            # print(a)
            # print("max one:{};index {}".format(np.max(a),np.argmax(a)))
        self.q_network_local_known.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})

        return action

    def update_memory_batch_errors(self, tree_idx, td_errors, rewards, is_unknown=True):
        loss_each_item = torch.abs(td_errors)
        loss_reward_each_item = loss_each_item + rewards
        loss_reward_each_item = loss_reward_each_item.detach().cpu().numpy()
        tree_idx = tree_idx[:, np.newaxis]
        if is_unknown:
            self.memory.batch_update_unknown(tree_idx, loss_reward_each_item)
        else:
            self.memory.batch_update_known(tree_idx, loss_reward_each_item)

    def compute_loss_and_td_errors(self, q_network_local, q_network_target, states, next_states, rewards, actions,
                                   dones, importance_sampling_weights):
        """Calculates the loss for the local Q network. It weighs each observations loss according to the importance
        sampling weights which come from the prioritised replay buffer"""
        Q_targets = self.compute_q_targets_d(q_network_local, q_network_target, next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values_d(q_network_local, states, actions)
        # loss = F.mse_loss(Q_expected, Q_targets)
        loss = self.weighted_mse_loss(Q_expected, Q_targets, importance_sampling_weights)
        # loss = torch.mean(loss)
        td_errors = Q_targets - Q_expected
        return loss, td_errors

    def weighted_mse_loss(self, input, target, weight):
        return torch.sum(weight * (input - target) ** 2)

    def compute_q_targets_d(self, q_network_local, q_network_target, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to trainer_p3d the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states_d(q_network_local, q_network_target, next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_expected_q_values_d(self, q_network_local, states, actions):
        """Computes the expected q_values we will use to create the loss to trainer_p3d the Q network"""
        Q_expected = q_network_local(states).gather(1,
                                                    actions.long())  # must convert actions to long so can be used as index
        return Q_expected

    def compute_q_values_for_next_states_d(self, q_network_local, q_network_target, next_states):
        """Computes the q_values for next state we will use to create the loss to trainer_p3d the Q network. Double DQN
        uses the local index to pick the maximum q_value action and then the target network to calculate the q_value.
        The reasoning behind this is that it will help stop the network from overestimating q values"""
        max_action_indexes = q_network_local(next_states).detach().argmax(1)
        Q_targets_next = q_network_target(next_states).gather(1, max_action_indexes.unsqueeze(1))
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to trainer_p3d the Q network"""
        Q_targets_current = rewards + (self.hyper_parameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def __build_model_and_optimizer_dict(self):
        model_dict = {"q_network_local_unknown": self.q_network_local_unknown,
                      "q_network_target_unknown": self.q_network_target_unknown,
                      "q_network_local_known": self.q_network_local_known,
                      "q_network_target_known": self.q_network_target_known
                      }
        optimizer_dict = {"q_network_optimizer_unknown", self.q_network_optimizer_unknown,
                          "q_network_optimizer_known", self.q_network_optimizer_known}

        print("model_dict:", model_dict)
        return model_dict, optimizer_dict
