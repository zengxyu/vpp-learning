import torch
import torch.nn.functional as F
from agents.DQN_agents.DDQN import DDQN
from memory.replay_buffer import PriorityReplayBuffer
import numpy as np


class DDQN_With_Prioritised_Experience_Replay_Multiplier(DDQN):
    """A DQN agent with prioritised experience replay"""
    agent_name = "DDQN with Prioritised Replay Buffer and Multiplier"

    def __init__(self, config):
        DDQN.__init__(self, config)
        self.memory = PriorityReplayBuffer(buffer_size=self.hyper_parameters['buffer_size'],
                                           batch_size=self.hyper_parameters['batch_size'],
                                           device=self.device, is_discrete=True, seed=self.seed)

    def learn(self):
        """Runs a learning iteration for the Q network after sampling from the replay buffer in a prioritised way"""
        # sampled_experiences, importance_sampling_weights = self.memory.sample()
        tree_idx, minibatch, ISWeights = self.memory.sample(is_vpp=self.config.environment['is_vpp'])
        states, actions, rewards, next_states, dones = minibatch
        # states, actions, rewards, next_states, dones = sampled_experiences
        loss, td_errors = self.compute_loss_and_td_errors(states, next_states, rewards, actions, dones, ISWeights)
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss,
                                    self.hyper_parameters["gradient_clipping_norm"])

        self.update_memory_batch_errors(tree_idx, td_errors, rewards)
        # self.soft_update_of_target_network(self.q_network_local, self.q_network_target, self.hyperparameters["tau"])
        self.skipping_step_update_of_target_network(self.q_network_local, self.q_network_target,
                                                    global_step_number=self.global_step_number,
                                                    update_every_n_steps=self.hyper_parameters["update_every_n_steps"])
        return loss.detach().cpu().numpy()

    def update_memory_batch_errors(self, tree_idx, td_errors, rewards):
        loss_each_item = torch.abs(td_errors)
        loss_reward_each_item = loss_each_item + rewards
        loss_reward_each_item = loss_reward_each_item.detach().cpu().numpy()
        tree_idx = tree_idx[:, np.newaxis]

        self.memory.batch_update(tree_idx, loss_reward_each_item)

    def compute_loss_and_td_errors(self, states, next_states, rewards, actions, dones, importance_sampling_weights):
        """Calculates the loss for the local Q network. It weighs each observations loss according to the importance
        sampling weights which come from the prioritised replay buffer"""
        Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        # loss = F.mse_loss(Q_expected, Q_targets)

        loss = self.weighted_mse_loss(Q_expected, Q_targets, importance_sampling_weights)
        # loss = torch.mean(loss)
        td_errors = Q_targets - Q_expected
        return loss, td_errors

    def weighted_mse_loss(self, input, target, weight):
        return torch.sum(weight * (input - target) ** 2)
