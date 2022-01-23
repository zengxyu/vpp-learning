import math

from rl_agents.exploration_strategies.base_exploration_strategy import BaseExplorationStrategy
import numpy as np
import random
import torch


class EpsilonGreedyExploration(BaseExplorationStrategy):
    """Implements an epsilon greedy exploration strategy"""

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        self.eps_exploration_strategy = self.hyperparameters['eps_exploration_strategy']
        print("Using exploration strategy : {}".format(self.eps_exploration_strategy))

        if "random_episodes_to_run" in self.hyperparameters.keys():
            self.random_episodes_to_run = self.hyperparameters["random_episodes_to_run"]
            print("Running {} random episodes".format(self.random_episodes_to_run))
        else:
            self.random_episodes_to_run = 0

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action_values = action_info["action_values"]
        turn_off_exploration = action_info["turn_off_exploration"]
        episode_number = action_info["episode_number"]
        global_step_number = action_info["global_step_number"]

        if turn_off_exploration:
            print(" ")
            print("Exploration has been turned OFF")
            print(" ")

        epsilon = self.get_updated_epsilon_exploration(action_info)

        # if episode_number % 50 == 0:
        #     print("Epsilon = {}".format(epsilon))

        if (random.random() > epsilon or turn_off_exploration) and (episode_number >= self.random_episodes_to_run):
            return torch.argmax(action_values).item()
        return np.random.randint(0, action_values.shape[1])

    def get_updated_epsilon_exploration(self, action_info):
        """Gets the probability that we just pick a random action. This probability decays the more episodes we have
        seen """

        episode_number = action_info["episode_number"]
        global_step_number = action_info["global_step_number"]

        if self.eps_exploration_strategy == "INVERSE_STRATEGY":
            epsilon = calculate_epsilon_with_inverse_strategy(
                self.hyperparameters['eps_exploration_strategy_params']['epsilon'], episode_number,
                self.hyperparameters['epsilon_decay_denominator'])
        elif self.eps_exploration_strategy == "EXPONENT_STRATEGY":
            epsilon = calculate_epsilon_with_exponent_strategy(
                self.hyperparameters['eps_exploration_strategy_params']['epsilon'], global_step_number,
                self.hyperparameters['eps_exploration_strategy_params']['epsilon_decay_rate'],
                self.hyperparameters['eps_exploration_strategy_params']['epsilon_min'])
        elif self.eps_exploration_strategy == "CYCLICAL_STRATEGY":
            epsilon = calculate_epsilon_with_cyclical_strategy(episode_number,
                                                               self.hyperparameters['eps_exploration_strategy_params'][
                                                                   "exploration_cycle_episodes_length"])
        else:
            raise NotImplementedError

        return epsilon

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        return reward_info["reward"]

    def reset(self):
        """Resets the noise process"""
        pass


def calculate_epsilon_with_cyclical_strategy(episode_number, exploration_cycle_episodes_length):
    """Calculates epsilon according to a cyclical strategy"""
    max_epsilon = 0.5
    min_epsilon = 0.001

    increment = (max_epsilon - min_epsilon) / float(exploration_cycle_episodes_length / 2)
    cycle = [ix for ix in range(int(exploration_cycle_episodes_length / 2))] + [ix for ix in range(
        int(exploration_cycle_episodes_length / 2), 0, -1)]
    cycle_ix = episode_number % exploration_cycle_episodes_length
    epsilon = max_epsilon - cycle[cycle_ix] * increment
    return epsilon


def calculate_epsilon_with_exponent_strategy(epsilon, global_step_number, epsilon_decay, epsilon_min):
    """Calculate epsilon according to an exponent of episode_number strategy"""
    # epsilon = epsilon / (1.0 + (episode_number / 1))
    epsilon = epsilon_min + (0.9 - epsilon_min) * (math.exp(-1.0 * (global_step_number + 1) / 200))
    # self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-1.0 * self.steps / self.decay))
    # epsilon = max(epsilon * epsilon_decay ** episode_number, epsilon_min)
    # print(episode_number, epsilon)
    return epsilon


def calculate_epsilon_with_inverse_strategy(epsilon, episode_number, epsilon_decay_denominator):
    """Calculates epsilon according to an inverse of episode_number strategy"""
    epsilon = epsilon / (1.0 + (episode_number / epsilon_decay_denominator))
    return epsilon
