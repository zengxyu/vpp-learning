import torch
import torch.nn.functional as functional
from torch import optim
from rl_agents.agents.base_agent import BaseAgent
from rl_agents.agents.base_agent_ac import BaseAgentAC
from rl_agents.exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration
from rl_agents.memory.replay_buffer import ReplayBuffer


class AgentDDPG(BaseAgentAC):
    """A AgentDDPG Agent"""
    agent_name = "AgentDDPG"

    def __init__(self, config, network_cls, action_space):
        BaseAgentAC.__init__(self, config, network_cls, action_space)
        self.action_size = action_space.size
        self.critic_local = self.create_critic_network(action_dim=self.action_size)
        self.critic_target = self.create_critic_network(action_dim=self.action_size)
        BaseAgent.copy_model_over(self.critic_local, self.critic_target)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.memory = ReplayBuffer(buffer_size=self.hyperparameters['buffer_size'],
                                   batch_size=self.hyperparameters['batch_size'],
                                   device=self.device, seed=self.seed)
        self.actor_local = self.create_actor_network(action_dim=self.action_size)
        self.actor_target = self.create_actor_network(action_dim=self.action_size)
        BaseAgent.copy_model_over(self.actor_local, self.actor_target)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.exploration_strategy = OU_Noise_Exploration(self.hyperparameters, self.action_size, self.seed)
        # self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
        #                       self.hyperparameters["theta"], self.hyperparameters["sigma"])
        self.model_dict, self.optimizer_dict = self.__build_model_and_optimizer_dict()

    def learn(self):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample()
        critic_loss = self.critic_learn(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        actor_loss = self.actor_learn(state_batch)
        return critic_loss.detach().cpu().numpy()

    def pick_action(self, state):
        """Picks an action using the actor network and then adds some noise to it to ensure exploration"""
        if isinstance(state, list):
            frame, robot_pose = state
            state = [torch.Tensor([frame]).to(self.device), torch.Tensor([robot_pose]).to(self.device)]
        else:
            state = torch.FloatTensor([state]).to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.is_train()
        # action += self.noise.sample()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action": action})

        return action.squeeze(0)

    def critic_learn(self, states, actions, rewards, next_states, dones):
        """Runs a learning iteration for the critic"""
        critic_loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.hyperparameters["Critic"]["tau"])
        return critic_loss

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss for the critic"""
        with torch.no_grad():
            critic_targets = self.compute_critic_targets(next_states, rewards, dones)
        critic_expected = self.compute_expected_critic_values(states, actions)
        loss = functional.mse_loss(critic_expected, critic_targets)
        return loss

    def compute_critic_targets(self, next_states, rewards, dones):
        """Computes the critic target values to be used in the loss for the critic"""
        critic_targets_next = self.compute_critic_values_for_next_states(next_states)
        critic_targets = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones)
        return critic_targets

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            critic_targets_next = self.critic_target(next_states, actions_next)
        return critic_targets_next

    def compute_critic_values_for_current_states(self, rewards, critic_targets_next, dones):
        """Computes the critic values for current states to be used in the loss for the critic"""
        critic_targets_current = rewards + (self.hyperparameters["discount_rate"] * critic_targets_next * (1.0 - dones))
        return critic_targets_current

    def compute_expected_critic_values(self, states, actions):
        """Computes the expected critic values to be used in the loss for the critic"""
        critic_expected = self.critic_local(states, actions)
        return critic_expected

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters[
            "update_every_n_steps"] == 0

    def actor_learn(self, states):
        """Runs a learning iteration for the actor"""
        actor_loss = self.calculate_actor_loss(states)
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.actor_local, self.actor_target, self.hyperparameters["Actor"]["tau"])
        return actor_loss

    def calculate_actor_loss(self, states):
        """Calculates the loss for the actor"""
        actions_pred = self.actor_local(states)
        q_v = self.critic_local(states, actions_pred)
        actor_loss = -q_v.mean()
        return actor_loss

    def __build_model_and_optimizer_dict(self):
        model_dict = {"critic_local": self.critic_local,
                      "critic_target": self.critic_target,
                      "actor_local": self.actor_local,
                      "actor_target": self.actor_target
                      }
        optimizer_dict = {"critic_optimizer", self.critic_optimizer,
                          "actor_optimizer", self.actor_optimizer
                          }
        return model_dict, optimizer_dict
