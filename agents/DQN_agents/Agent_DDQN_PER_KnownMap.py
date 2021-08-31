import torch
from torch import optim

from agents.Base_Agent_DQN import Base_Agent_DQN
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
import numpy as np

from memory.replay_buffer_knownmap import PriorityReplayBufferKnownMap


class Agent_DDQN_PER_KnownMap(Base_Agent_DQN):
    """A DQN agent with prioritised experience replay"""
    agent_name = "DDQN with Prioritised Replay"

    def __init__(self, config):
        Base_Agent_DQN.__init__(self, config)
        self.q_network_local = self.create_NN()
        self.q_network_target = self.create_NN()
        Base_Agent_DQN.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyper_parameters["learning_rate"], eps=1e-4)
        self.exploration_strategy = Epsilon_Greedy_Exploration(self.hyper_parameters)

        self.memory = PriorityReplayBufferKnownMap(buffer_size=self.hyper_parameters['buffer_size'],
                                                   batch_size=self.hyper_parameters['batch_size'],
                                                   device=self.device, is_discrete=True, seed=self.seed)
        print("load replay buffer from local")
        self.model_dict, self.optimizer_dict = self.__build_model_and_optimizer_dict()

    def pick_action(self, state):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        global_map, frame, robot_pose = state
        state = [torch.Tensor([global_map]).to(self.device), torch.Tensor([frame]).to(self.device),
                 torch.Tensor([robot_pose]).to(self.device)]

        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})

        return action

    def learn(self):
        tree_idx, minibatch, ISWeights = self.memory.sample(is_vpp=self.config.environment['is_vpp'])
        states, actions, rewards, next_states, dones = minibatch

        loss_dqn, td_errors = self.compute_loss_and_td_errors(states, next_states, rewards, actions, dones, ISWeights)

        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss_dqn,
                                    self.hyper_parameters["gradient_clipping_norm"])

        self.update_memory_batch_errors(tree_idx, td_errors, rewards)
        self.skipping_step_update_of_target_network(self.q_network_local, self.q_network_target,
                                                    global_step_number=self.global_step_number,
                                                    update_every_n_steps=self.hyper_parameters["update_every_n_steps"])
        return loss_dqn.detach().cpu().numpy()

    def update_memory_batch_errors(self, tree_idx, td_errors, rewards):
        loss_each_item = torch.abs(td_errors)
        loss_reward_each_item = loss_each_item + rewards
        loss_reward_each_item = loss_reward_each_item.detach().cpu().numpy()
        tree_idx = tree_idx[:, np.newaxis]

        self.memory.batch_update(tree_idx, loss_reward_each_item)

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to trainer_p3d the Q network. Double DQN
        uses the local index to pick the maximum q_value action and then the target network to calculate the q_value.
        The reasoning behind this is that it will help stop the network from overestimating q values"""
        max_action_indexes = self.q_network_local(next_states).detach().argmax(1)
        a = self.q_network_target(next_states)
        Q_targets_next = a.gather(1, max_action_indexes.unsqueeze(1))
        return Q_targets_next

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

    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to trainer_p3d the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to trainer_p3d the Q network"""
        Q_targets_current = rewards + (self.hyper_parameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to trainer_p3d the Q network"""
        Q_expected = self.q_network_local(states).gather(1,
                                                            actions.long())  # must convert actions to long so can be used as index
        return Q_expected

    def __build_model_and_optimizer_dict(self):
        model_dict = {"q_network_local": self.q_network_local,
                      "q_network_target": self.q_network_target}
        optimizer_dict = {"q_network_optimizer", self.q_network_optimizer}
        print("model_dict:", model_dict)
        return model_dict, optimizer_dict
