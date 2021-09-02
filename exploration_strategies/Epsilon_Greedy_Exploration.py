from exploration_strategies.Base_Exploration_Strategy import Base_Exploration_Strategy
from utilities.data_structures.Constant import *
import numpy as np
import random
import torch


class Epsilon_Greedy_Exploration(Base_Exploration_Strategy):
    """Implements an epsilon greedy exploration strategy"""

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        self.eps_exploration_strategy = self.hyperparameters['eps_exploration_strategy']
        print("Using exploration strategy : {}".format(
            EpsExplorationStrategies.get(self.eps_exploration_strategy)))

        if "random_episodes_to_run" in self.hyperparameters.keys():
            self.random_episodes_to_run = self.hyperparameters["random_episodes_to_run"]
            print("Running {} random episodes".format(self.random_episodes_to_run))
        else:
            self.random_episodes_to_run = 0

        self.is_exploration_hyperparameters_valid()

    def is_exploration_hyperparameters_valid(self):
        """validate whether the hyperparameters are valid"""
        if self.eps_exploration_strategy == EpsExplorationStrategy.INVERSE_STRATEGY:
            assert "epsilon" in self.hyperparameters.keys()
            assert "epsilon_decay_denominator" in self.hyperparameters.keys()
        elif self.eps_exploration_strategy == EpsExplorationStrategy.EXPONENT_STRATEGY:
            assert "epsilon" in self.hyperparameters.keys()
            assert "epsilon_decay_rate" in self.hyperparameters.keys()
            assert "epsilon_min" in self.hyperparameters.keys()
        else:
            assert "exploration_cycle_episodes_length" in self.hyperparameters.keys()

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action_values = action_info["action_values"]
        turn_off_exploration = action_info["turn_off_exploration"]
        episode_number = action_info["episode_number"]

        if turn_off_exploration:
            print(" ")
            print("Exploration has been turned OFF")
            print(" ")

        epsilon = self.get_updated_epsilon_exploration(action_info)

        if episode_number % 50 == 0:
            print("Epsilon = {}".format(epsilon))

        if (random.random() > epsilon or turn_off_exploration) and (episode_number >= self.random_episodes_to_run):
            # print("choose the best action: --", torch.argmax(action_values).item())
            return torch.argmax(action_values).item()
        random_action = np.random.randint(0, action_values.shape[1])
        # print("randomly choose action:", random_action)
        return random_action

    def get_updated_epsilon_exploration(self, action_info):
        """Gets the probability that we just pick a random action. This probability decays the more episodes we have
        seen """

        episode_number = action_info["episode_number"]

        if self.eps_exploration_strategy == EpsExplorationStrategy.INVERSE_STRATEGY:
            epsilon = calculate_epsilon_with_inverse_strategy(self.hyperparameters['epsilon'], episode_number,
                                                              self.hyperparameters['epsilon_decay_denominator'])
        elif self.eps_exploration_strategy == EpsExplorationStrategy.EXPONENT_STRATEGY:
            epsilon = calculate_epsilon_with_exponent_strategy(self.hyperparameters['epsilon'], episode_number,
                                                               self.hyperparameters['epsilon_decay_rate'],
                                                               self.hyperparameters['epsilon_min'])
        else:
            epsilon = calculate_epsilon_with_cyclical_strategy(episode_number, self.hyperparameters[
                "exploration_cycle_episodes_length"])

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


def calculate_epsilon_with_exponent_strategy(epsilon, episode_number, epsilon_decay, epsilon_min):
    """Calculate epsilon according to an exponent of episode_number strategy"""
    epsilon = max(epsilon * epsilon_decay ** episode_number, epsilon_min)
    return epsilon


def calculate_epsilon_with_inverse_strategy(epsilon, episode_number, epsilon_decay_denominator):
    """Calculates epsilon according to an inverse of episode_number strategy"""
    epsilon = epsilon / (1.0 + (episode_number / epsilon_decay_denominator))
    return epsilon
