from collections import Counter

import torch
import torch.optim as optim
import torch.nn.functional as F
from agents.Base_Agent_DQN import Base_Agent_DQN
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from memory.replay_buffer import ReplayBuffer


class DQN(Base_Agent_DQN):
    """A deep Q learning agent"""
    agent_name = "DQN"

    def __init__(self, config):
        Base_Agent_DQN.__init__(self, config)
        self.memory = ReplayBuffer(buffer_size=self.hyper_parameters['buffer_size'],
                                   batch_size=self.hyper_parameters['batch_size'],
                                   device=self.device, seed=self.seed)
        self.q_network_local = self.create_NN()
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyper_parameters["learning_rate"], eps=1e-4)
        self.exploration_strategy = Epsilon_Greedy_Exploration(self.hyper_parameters)
        self.model_dict, self.optimizer_dict = self.__build_model_and_optimizer_dict()

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

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        if experiences is None:
            states, actions, rewards, next_states, dones = self.memory.sample()  # Sample experiences
        else:
            states, actions, rewards, next_states, dones = experiences
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
        output = self.q_network_local(next_states).detach()
        output1 = output[:, 0:12]
        output2 = output[:, 12:15]

        Q_targets_next1 = output1.max(1)[0].unsqueeze(1)
        Q_targets_next2 = output2.max(1)[0].unsqueeze(1)

        return Q_targets_next1 + Q_targets_next2

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        Q_targets_current = rewards + (self.hyper_parameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        actions1 = actions[:, 0:12]
        actions2 = actions[:, 12:]
        output = self.q_network_local(states)
        Q_expected1 = output.gather(1, actions1.long())  # must convert actions to long so can be used as index
        Q_expected2 = output.gather(1, actions2.long())  # must convert actions to long so can be used as index
        return Q_expected1 + Q_expected2

    def locally_save_policy(self):
        """Saves the policy"""
        torch.save(self.q_network_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def __build_model_and_optimizer_dict(self):
        model_dict = {"q_network_local": self.q_network_local}
        optimizer_dict = {"q_network_optimizer", self.q_network_optimizer}
        return model_dict, optimizer_dict
