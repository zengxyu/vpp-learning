import torch
import torch.nn.functional as functional
from torch import optim
from agents.Base_Agent import Base_Agent
from utilities.OU_Noise import OU_Noise
from .DDPG import DDPG
from exploration_strategies.Gaussian_Exploration import Gaussian_Exploration


class TD3(DDPG):
    """A TD3 Agent from the paper Addressing Function Approximation Error in Actor-Critic Methods (Fujimoto et al. 2018)
    https://arxiv.org/abs/1802.09477
    Twin Delayed Deep Deterministic Policy Gradient
    1. 在TD3中，使用 两套网络(Twin) 表示不同的Q值，通过选取最小的那个作为我们更新的目标（Target Q Value），抑制持续地过高估计。 ——TD3的基本思路
    2. 这里说的Dalayed ，是Actor更新的Delay。 也就是说相对于Critic可以更新多次后，Actor再进行更新。
    3. TD3在目标网络估计Expected Return部分，对policy网络引入随机噪声

    """
    agent_name = "TD3"

    def __init__(self, config):
        DDPG.__init__(self, config)
        self.critic_local_2 = self.create_critic_network(state_dim=self.state_size,
                                                         action_dim=self.action_size, output_dim=1)
        self.critic_target_2 = self.create_critic_network(state_dim=self.state_size,
                                                          action_dim=self.action_size, output_dim=1)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(),
                                             lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.exploration_strategy_critic = Gaussian_Exploration(self.hyperparameters)


    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            actions_next_with_noise = self.exploration_strategy_critic.perturb_action_for_exploration_purposes(
                {"action": actions_next})
            critic_targets_next_1 = self.critic_target(next_states, actions_next_with_noise)
            critic_targets_next_2 = self.critic_target_2(next_states, actions_next_with_noise)
            critic_targets_next = torch.min(torch.cat((critic_targets_next_1, critic_targets_next_2), 1), dim=1)[
                0].unsqueeze(-1)
        return critic_targets_next

    def critic_learn(self, states, actions, rewards, next_states, dones):
        """Runs a learning iteration for both the critics"""
        critic_targets_next = self.compute_critic_values_for_next_states(next_states)
        critic_targets = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones)

        critic_expected_1 = self.critic_local(states, actions)
        critic_expected_2 = self.critic_local_2(states, actions)

        critic_loss_1 = functional.mse_loss(critic_expected_1, critic_targets)
        critic_loss_2 = functional.mse_loss(critic_expected_2, critic_targets)

        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])

        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

        return critic_loss_1
