from collections import Counter

import torch
import torch.optim as optim
import torch.nn.functional as F
from rl_agents.agents.base_agent_dqn import BaseAgentDQN
from rl_agents.exploration_strategies.epsilon_greedy_exploration import EpsilonGreedyExploration
from rl_agents.memory.memory_optimized_replay_buffer import MemoryOptimizedReplayBuffer
from rl_agents.memory.replay_buffer import ReplayBuffer


class AgentDQN(BaseAgentDQN):
    """A deep Q learning agent"""
    agent_name = "DQN"

    def __init__(self, config, network_cls, action_space):
        BaseAgentDQN.__init__(self, config, network_cls, action_space)
        self.memory = ReplayBuffer(buffer_size=self.hyper_parameters['buffer_size'],
                                   batch_size=self.hyper_parameters['batch_size'],
                                   device=self.device, seed=self.seed)
        # self.memory = MemoryOptimizedReplayBuffer(buffer_size=self.hyper_parameters['buffer_size'],
        #                                           batch_size=self.hyper_parameters['batch_size'],
        #                                           history_length=self.hyper_parameters["history_length"],
        #                                           device=self.device,
        #                                           seed=self.seed)
        self.q_network_local = self.create_network()
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyper_parameters["learning_rate"], eps=1e-4)
        self.exploration_strategy = EpsilonGreedyExploration(self.hyper_parameters)
        self.model_dict, self.optimizer_dict = self.__build_model_and_optimizer_dict()

    def pick_action(self, state):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        state = [torch.Tensor([obs]).to(self.device) for obs in state]
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number,
                                                                                    "global_step_number": self.global_step_number

                                                                                    })
        return action

    def learn(self):
        """Runs a learning iteration for the Q network"""
        if not self.time_for_q_network_to_learn():
            return 0

        self.q_network_local.is_train()
        states, actions, rewards, next_states, dones = self.memory.sample()  # Sample experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)

        actions_list = [action_X.item() for action_X in actions]

        self.logger.info("Action counts {}".format(Counter(actions_list)))
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss,
                                    self.hyper_parameters["gradient_clipping_norm"])

        return loss.detach().cpu().numpy()

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
        Q_targets_current = rewards + (self.hyper_parameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        Q_expected = self.q_network_local(states).gather(1,
                                                         actions.long())  # must convert action_space to long so can be used as index
        return Q_expected

    def locally_save_policy(self):
        """Saves the policy"""
        torch.save(self.q_network_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def __build_model_and_optimizer_dict(self):
        model_dict = {"q_network_local": self.q_network_local}
        optimizer_dict = {"q_network_optimizer", self.q_network_optimizer}
        return model_dict, optimizer_dict
