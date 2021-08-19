import pickle
import os
import torch
import torch.nn.functional as F
from torch import optim

from agents.Base_Agent_DQN import Base_Agent_DQN
from agents.DQN_agents.DDQN import DDQN
from agents.VAE_Learner import VAE_Learner
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
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


def minmaxscaler(data):
    min = torch.min(data)
    max = torch.max(data)
    return (data - min) / (max - min)


class Agent_DDQN_PER_VAE(Base_Agent_DQN):
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

        self.memory = PriorityReplayBuffer(buffer_size=self.hyper_parameters['buffer_size'],
                                           batch_size=self.hyper_parameters['batch_size'],
                                           device=self.device, is_discrete=True, seed=self.seed)
        self.encoder_parameter_names = ["encoder_conv_layer.0.weight", "encoder_conv_layer.0.bias",
                                        "encoder_linear_layer.0.weight", "encoder_linear_layer.0.bias",
                                        "fc_mu.weight", "fc_mu.bias",
                                        "fc_std.weight", "fc_std.bias"]
        self.decoder_parameter_names = ["decoder_linear_layer.0.weight", "decoder_linear_layer.0.bias",
                                        "decoder_linear_layer.2.weight", "decoder_linear_layer.2.bias",
                                        "decoder_conv_layer.0.weight", "decoder_conv_layer.0.bias"
                                        ]
        self.dqn_parameter_names = ["pose_fc1a.weight", "pose_fc1a.bias",
                                    "pose_fc2a.weight", "pose_fc2a.bias",
                                    "pose_fc1b.weight", "pose_fc1b.bias",
                                    "pose_fc2b.weight", "pose_fc2b.bias",
                                    "pose_fc3.weight", "pose_fc3.bias",
                                    "pose_fc4.weight", "pose_fc4.bias",
                                    "fc_val.weight", "fc_val.bias"
                                    ]
        # self.memory_old = pickle.load(open(os.path.join(self.config.folder['exp_in'], "buffer.obj"), 'rb'))
        print("load replay buffer from local")
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

        state[0] = minmaxscaler(state[0])
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)[0]
        self.q_network_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})

        return action

    def pre_learn_vae(self, num=3000, resume=True):
        # from_model_local_path = os.path.join(self.config.folder['model_in'], "Agent_q_network_local_51.mdl")
        # from_model_target_path = os.path.join(self.config.folder['model_in'], "Agent_q_network_target_51.mdl")
        #
        # from_model_local_dict = torch.load(from_model_local_path, map_location=self.device)
        # from_model_target_dict = torch.load(from_model_target_path, map_location=self.device)
        #
        # self.copy_encoder_model_over(from_model_local_dict, self.q_network_local)
        # self.copy_encoder_model_over(from_model_target_dict, self.q_network_target)

        self.fix_dqn_parameters()
        print("fixed dqn parameters")
        self.check_fixed_parameters()
        # self.check_fixed_dqn_parameters()
        for i in range(num):
            tree_idx, minibatch, ISWeights = self.memory_old.sample(is_vpp=self.config.environment['is_vpp'],
                                                                    is_weighted=False)
            states, actions, rewards, next_states, dones = minibatch
            states[0] = minmaxscaler(states[0])
            # compute vae loss
            loss_vae = self.compute_vae_loss(states)

            self.take_optimisation_step_vae_only(self.q_network_optimizer, self.q_network_local, loss_vae)

            if i % 100 == 0:
                frame = states[0]
                # print("max value:{}; min value:{}".format(torch.max(frame), torch.min(frame)))
                print("i:{}; loss_vae:{}".format(i, loss_vae))
        self.unfixed_dqn_parameters()
        print("unfixed dqn parameters")
        # self.check_fixed_dqn_parameters()

    def learn_vae(self):
        # 训练vae
        # sampled_experiences, importance_sampling_weights = self.memory.sample()
        tree_idx, minibatch, ISWeights = self.memory_old.sample(is_vpp=self.config.environment['is_vpp'],
                                                                is_weighted=False)
        states, actions, rewards, next_states, dones = minibatch
        states[0] = minmaxscaler(states[0])
        # compute vae loss
        loss_vae = self.compute_vae_loss(states)
        print("loss_vae:{}".format(loss_vae))
        self.take_optimisation_step_vae_only(self.q_network_optimizer, self.q_network_local, loss_vae)
        return loss_vae, states

    def take_optimisation_step_vae_only(self, q_network_optimizer, q_network_local, loss_vae):
        q_network_optimizer.zero_grad()  # reset gradients to 0
        loss_vae.backward()  # this calculates the gradients
        q_network_optimizer.step()

    def learn_dqn(self):
        tree_idx, minibatch, ISWeights = self.memory.sample(is_vpp=self.config.environment['is_vpp'])
        states, actions, rewards, next_states, dones = minibatch
        states[0] = minmaxscaler(states[0])
        next_states[0] = minmaxscaler(next_states[0])
        loss_dqn, td_errors = self.compute_loss_and_td_errors(states, next_states, rewards, actions, dones, ISWeights)
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss_dqn,
                                    self.hyper_parameters["gradient_clipping_norm"])

        self.update_memory_batch_errors(tree_idx, td_errors, rewards)
        self.skipping_step_update_of_target_network(self.q_network_local, self.q_network_target,
                                                    global_step_number=self.global_step_number,
                                                    update_every_n_steps=self.hyper_parameters["update_every_n_steps"])
        return loss_dqn

    # def learn(self):
    #     """Runs a learning iteration for the Q network after sampling from the replay buffer in a prioritised way"""
    #     """
    #     (1).用之前训练的experience来训练autoencoder，稳定后，fix参数训练dqn模型
    #     (2).dqn模型训练稳定后
    #     (3).放开autoencoder参数，进行refine训练
    #     """
    #     # 先训练100次vae,
    #     # 然后固定编码层的参数，再训练dqn,
    #     # 然后解开编码层，又训练100次vae
    #
    #     # # 先训练100次vae,
    #     # 解开编码层, 固定DQN的参数
    #
    #     # 再训练dqn,
    #     loss_dqn = self.learn_dqn()
    #
    #     # if self.global_step_number % (5 * 300) == 0:
    #     #     print("训练VAE")
    #     #     self.unfixed_encoder_parameters()
    #     #     self.fix_dqn_parameters()
    #     #     self.check_fixed_parameters()
    #     #     vae_steps = 10
    #     #     for i in range(vae_steps):
    #     #         self.learn_vae()
    #     #     # 固定编码层的参数, 解开DQN的参数
    #     #     self.fix_encoder_parameters()
    #     #     self.unfixed_dqn_parameters()
    #     #     self.check_fixed_parameters()
    #     if self.episode_number >= 50:
    #         self.fix_decoder_parameters()
    #         self.unfixed_encoder_parameters()
    #         self.unfixed_dqn_parameters()
    #     return loss_dqn.detach().cpu().numpy()
    def learn(self):
        tree_idx, minibatch, ISWeights = self.memory.sample(is_vpp=self.config.environment['is_vpp'])
        states, actions, rewards, next_states, dones = minibatch
        states[0] = minmaxscaler(states[0])
        next_states[0] = minmaxscaler(next_states[0])
        loss_vae = self.compute_vae_loss(states, next_states, actions)

        loss_dqn, td_errors = self.compute_loss_and_td_errors(states, next_states, rewards, actions, dones, ISWeights)

        if self.global_step_number % 50 == 0:
            print("loss vae:{};loss dqn:{}".format(loss_vae, loss_dqn))
        loss = loss_vae + loss_dqn
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss,
                                    self.hyper_parameters["gradient_clipping_norm"])

        self.update_memory_batch_errors(tree_idx, td_errors, rewards)
        self.skipping_step_update_of_target_network(self.q_network_local, self.q_network_target,
                                                    global_step_number=self.global_step_number,
                                                    update_every_n_steps=self.hyper_parameters["update_every_n_steps"])
        return loss.detach().cpu().numpy()

    def imitation_learning(self):
        pass

    def update_memory_batch_errors(self, tree_idx, td_errors, rewards):
        loss_each_item = torch.abs(td_errors)
        loss_reward_each_item = loss_each_item + rewards
        loss_reward_each_item = loss_reward_each_item.detach().cpu().numpy()
        tree_idx = tree_idx[:, np.newaxis]

        self.memory.batch_update(tree_idx, loss_reward_each_item)

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network. Double DQN
        uses the local index to pick the maximum q_value action and then the target network to calculate the q_value.
        The reasoning behind this is that it will help stop the network from overestimating q values"""
        max_action_indexes = self.q_network_local(next_states)[0].detach().argmax(1)
        a = self.q_network_target(next_states)[0]
        Q_targets_next = a.gather(1, max_action_indexes.unsqueeze(1))
        return Q_targets_next

    def compute_vae_loss(self, states, next_states, actions):
        # out_decoder size : 128 * 12 * 15 * 36 * 18
        _, out_decoder, mu, logvar = self.q_network_local(states)
        batch_size = out_decoder.shape[0]
        actions = actions.long()
        next_states_predictions = None
        for i in range(batch_size):
            next_states_prediction = out_decoder[i][actions[i]]
            if next_states_predictions is None:
                next_states_predictions = next_states_prediction
            else:
                next_states_predictions = torch.cat([next_states_predictions, next_states_prediction], dim=0)
        # out_decoder = torch.transpose(out_decoder, 0, 1)
        #
        # next_states_predictions = out_decoder[actions]
        loss_vae = loss_function(next_states_predictions, next_states[0], mu, logvar) / states[0].size(0)
        return loss_vae

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
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        Q_targets_current = rewards + (self.hyper_parameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        Q_expected = self.q_network_local(states)[0].gather(1,
                                                            actions.long())  # must convert actions to long so can be used as index
        return Q_expected

    def __build_model_and_optimizer_dict(self):
        model_dict = {"q_network_local": self.q_network_local,
                      "q_network_target": self.q_network_target}
        optimizer_dict = {"q_network_optimizer", self.q_network_optimizer}
        print("model_dict:", model_dict)
        return model_dict, optimizer_dict

    def check_fixed_parameters(self):
        for k, v in self.q_network_local.named_parameters():
            print(k, ":", v.requires_grad)

    def fix_encoder_parameters(self):
        for k, v in self.q_network_local.named_parameters():
            if k in self.encoder_parameter_names:
                v.requires_grad = False  # 固定参数

    def unfixed_encoder_parameters(self):
        for k, v in self.q_network_local.named_parameters():
            if k in self.encoder_parameter_names:
                v.requires_grad = True  # 解开参数

    def fix_decoder_parameters(self):
        for k, v in self.q_network_local.named_parameters():
            if k in self.decoder_parameter_names:
                v.requires_grad = False  # 固定参数

    def unfixed_decoder_parameters(self):
        for k, v in self.q_network_local.named_parameters():
            if k in self.decoder_parameter_names:
                v.requires_grad = True  # 解开参数

    # def check_fixed_dqn_parameters(self):
    #     for k, v in self.q_network_local.named_parameters():
    #         if k in self.dqn_parameter_names:
    #             print(k, ":", v.requires_grad)
    #             if not v.requires_grad:
    #                 return False
    #     return True

    def fix_dqn_parameters(self):
        for k, v in self.q_network_local.named_parameters():
            if k in self.dqn_parameter_names:
                v.requires_grad = False  # 固定参数

    def unfixed_dqn_parameters(self):
        for k, v in self.q_network_local.named_parameters():
            if k in self.dqn_parameter_names:
                v.requires_grad = True  # 解开参数

    def copy_encoder_model_over(self, from_model_dict, to_model):

        to_model_dict = to_model.state_dict()
        from_model_encoder_dict = {k: v for k, v in from_model_dict.items() if k in self.encoder_parameter_names}
        print(
            from_model_encoder_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        to_model_dict.update(from_model_encoder_dict)
        to_model.load_state_dict(to_model_dict)
