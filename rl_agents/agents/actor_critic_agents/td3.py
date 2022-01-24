import torch
import torch.nn.functional as functional
from torch import optim

from rl_agents.agents.actor_critic_agents.ddpg import AgentDDPG
from rl_agents.agents.base_agent import BaseAgent
from rl_agents.exploration_strategies.Gaussian_Exploration import Gaussian_Exploration


class TD3(AgentDDPG):
    """A TD3 Agent from the paper Addressing Function Approximation Error in Actor-Critic Methods (Fujimoto et al. 2018)
    https://arxiv.org/abs/1802.09477
    Twin Delayed Deep Deterministic Policy Gradient
    1. 在TD3中，使用 两套网络(Twin) 表示不同的Q值，通过选取最小的那个作为我们更新的目标（Target Q Value），抑制持续地过高估计。 ——TD3的基本思路
    2. 这里说的Dalayed ，是Actor更新的Delay。 也就是说相对于Critic可以更新多次后，Actor再进行更新。
    3. TD3在目标网络估计Expected Return部分，对policy网络引入随机噪声

    """
    agent_name = "TD3"

    def __init__(self, config, network_cls, action_space):
        AgentDDPG.__init__(self, config, network_cls, action_space)
        self.critic_local_2 = self.create_critic_network(action_dim=self.action_size)
        self.critic_target_2 = self.create_critic_network(action_dim=self.action_size)
        BaseAgent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(),
                                             lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.exploration_strategy_critic = Gaussian_Exploration(self.hyperparameters)
        self.model_dict, self.optimizer_dict = self.__build_model_and_optimizer_dict()

    def learn(self):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample()
        critic_loss = self.critic_learn(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        if self.global_step_number % 3 == 0:
            actor_loss = self.actor_learn(state_batch)
        return critic_loss.detach().cpu().numpy()

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            actions_next_with_noise = self.exploration_strategy_critic.perturb_action_for_exploration_purposes(
                {"action": actions_next})
            critic_targets_next_1 = self.critic_target(next_states, actions_next_with_noise)
            critic_targets_next_2 = self.critic_target_2(next_states, actions_next_with_noise)
            critic_targets_next = torch.min(critic_targets_next_1, critic_targets_next_2)
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

    def __build_model_and_optimizer_dict(self):
        model_dict = {"critic_local": self.critic_local,
                      "critic_target": self.critic_target,
                      "critic_local_2": self.critic_local_2,
                      "critic_target_2": self.critic_target_2,
                      "actor_local": self.actor_local,
                      "actor_target": self.actor_target,

                      }
        optimizer_dict = {"critic_optimizer", self.critic_optimizer,
                          "critic_optimizer_2", self.critic_optimizer_2,
                          "actor_optimizer", self.actor_optimizer
                          }
        return model_dict, optimizer_dict