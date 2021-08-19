import torch
import torch.nn.functional as F
from agents.DQN_agents.DDQN import DDQN
from agents.VAE_Learner import VAE_Learner
from memory.replay_buffer import PriorityReplayBuffer
import numpy as np

reconstruction_function = torch.nn.MSELoss(reduction='sum')


def loss_function(recon_x, x, mu, logvar):
    MSE = reconstruction_function(recon_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return MSE + KLD


class DDQN_PER_VAE(DDQN):
    """A DQN agent with prioritised experience replay"""
    agent_name = "DDQN with Prioritised Replay"

    def __init__(self, config):
        DDQN.__init__(self, config)
        self.memory = PriorityReplayBuffer(buffer_size=self.hyper_parameters['buffer_size'],
                                           batch_size=self.hyper_parameters['batch_size'],
                                           device=self.device, is_discrete=True, seed=self.seed)
        self.auto_encoder_learner = VAE_Learner()

    def VAE_learn(self, data):
        mu, logvar, predictions = self.model(data)
        loss = loss_function(predictions, data, mu, logvar) / data.size(0)

        loss.backward()
        self.optimizer.step()
        print(' - loss : ', str(loss.item()))
        return mu

    def learn(self):
        """Runs a learning iteration for the Q network after sampling from the replay buffer in a prioritised way"""
        # sampled_experiences, importance_sampling_weights = self.memory.sample()
        tree_idx, minibatch, ISWeights = self.memory.sample(is_vpp=self.config.environment['is_vpp'])
        states, actions, rewards, next_states, dones = minibatch
        frame_in = states[0]
        frame_in_next = next_states[0]
        code = self.auto_encoder_learner.learn(frame_in)
        code_next = self.auto_encoder_learner.learn(frame_in_next)
        states = [code, states[1]]
        next_states = [code_next, next_states[1]]

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
