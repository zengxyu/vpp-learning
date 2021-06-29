from collections import Counter

import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agents.Base_Agent_DQN import Base_Agent_DQN
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from memory.replay_buffer import ReplayBuffer


class DQN(Base_Agent_DQN):
    """A deep Q learning agent"""
    agent_name = "DQN"

    def __init__(self, config):
        Base_Agent_DQN.__init__(self, config)
        self.memory = ReplayBuffer(buffer_size=self.hyperparameters['buffer_size'],
                                   batch_size=self.hyperparameters['batch_size'],
                                   device=self.device, seed=self.seed)
        self.q_network_local = self.create_NN()
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"], eps=1e-4)
        self.exploration_strategy = Epsilon_Greedy_Exploration(self.hyperparameters)

    def step(self, state, action, reward, next_state, done):
        self.memory.add_experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.global_step_number += 1

    def pick_action(self, state):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if isinstance(state, list):
            frame, robot_pose = state
            state = [torch.Tensor([frame]).to(self.device), torch.Tensor([robot_pose]).to(self.device)]
        else:
            state = torch.FloatTensor([state]).to(self.device)

        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})

        return action

        # if state is None: state = self.state
        # if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
        # state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # if len(state.shape) < 2: state = state.unsqueeze(0)
        # self.q_network_local.eval()  # puts network in evaluation mode
        # with torch.no_grad():
        #     action_values = self.q_network_local(state)
        # self.q_network_local.train()  # puts network back in training mode
        # action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
        #                                                                             "turn_off_exploration": self.turn_off_exploration,
        #                                                                             "episode_number": self.episode_number})
        # self.logger.info("Q values {} -- Action chosen {}".format(action_values, action))
        # return action

    # def pick_action(self, state):
    #     if isinstance(state, list):
    #         # print("state is a list")
    #         frame, robot_pose = state
    #         rnd = random.random()
    #         self.eps = 0.5
    #         eps_decay = 0.99999
    #         eps_min = 0.15
    #         self.eps = max(self.eps * eps_decay, eps_min)
    #         if rnd < self.eps:
    #             return np.random.randint(self.hyperparameters['action_size'])
    #         else:
    #             frame_in = torch.Tensor([frame]).to(self.device)
    #             robot_pose_in = torch.Tensor([robot_pose]).to(self.device)
    #
    #             self.q_network_local.eval()
    #             with torch.no_grad():
    #                 q_val = self.q_network_local([frame_in, robot_pose_in])
    #             self.q_network_local.train()
    #             action = np.argmax(q_val.cpu().data.numpy())
    #         return action
    #     else:
    #         # print("state is not a list")
    #         rnd = random.random()
    #         self.eps = 0.5
    #         eps_decay = 0.99999
    #         eps_min = 0.15
    #         self.eps = max(self.eps * eps_decay, eps_min)
    #         if rnd < self.eps:
    #             return np.random.randint(self.hyperparameters['action_size'])
    #         else:
    #             state = torch.FloatTensor([state]).to(self.device)
    #             self.q_network_local.eval()
    #             with torch.no_grad():
    #                 q_val = self.q_network_local(state)
    #             self.q_network_local.train()
    #             action = np.argmax(q_val.cpu().data.numpy())
    #         # action = float(action)
    #         return action

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        if self.enough_experiences_to_learn_from():
            if experiences is None:
                states, actions, rewards, next_states, dones = self.memory.sample()  # Sample experiences
            else:
                states, actions, rewards, next_states, dones = experiences
            loss = self.compute_loss(states, next_states, rewards, actions, dones)

            actions_list = [action_X.item() for action_X in actions]

            self.logger.info("Action counts {}".format(Counter(actions_list)))
            self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss,
                                        self.hyperparameters["gradient_clipping_norm"])
            return loss.detach().cpu().numpy()
        return 0

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        """这里应该是q_network_target()"""

        Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        Q_expected = self.q_network_local(states).gather(1,
                                                         actions.long())  # must convert actions to long so can be used as index
        return Q_expected

    def locally_save_policy(self):
        """Saves the policy"""
        torch.save(self.q_network_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    # def reset_game(self):
    #     super(DQN, self).reset_game()
    #     self.update_learning_rate(self.hyperparameters["learning_rate"], self.q_network_optimizer)

    # def time_for_q_network_to_learn(self):
    #     """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
    #     enough experiences in the replay buffer to learn from"""
    #     return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()
    #
    # def right_amount_of_steps_taken(self):
    #     """Returns boolean indicating whether enough steps have been taken for learning to begin"""
    #     return self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0
