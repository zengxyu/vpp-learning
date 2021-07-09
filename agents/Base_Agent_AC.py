import random
import os
import logging
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class Base_Agent_AC:
    def __init__(self, config):
        self.name = "grid world"
        self.seed = random.seed(42)
        self.logger = self.setup_logger()
        self.config = config
        self.debug_mode = config.debug_mode
        self.hyperparameters = config.hyperparameters['Actor_Critic_Agents']

        self.device = torch.device("cuda") if torch.cuda.is_available() and config.use_GPU else torch.device("cpu")
        print("device:", self.device)

        # if params['resume']:
        #     print("resume model")
        #     self.load_model(net=self.q_network_local, file_path=params['model_path'], map_location=self.device)
        #     self.update_target_network()
        #     if not self.params['is_train']:
        #         # 如果不是训练状态的话，不更新
        #         self.update_every = 1000000000000000000
        # print(self.q_network_local)
        self.turn_off_exploration = False
        self.global_step_number = 0
        self.episode_number = 0

    # def create_NN(self):
    #     Model = self.config.model
    #     net = Model(action_size=self.hyperparameters['action_size']).to(self.device)
    #     return net

    # def create_NN(self, input_dim, output_dim, key_to_use=None, override_seed=None, hyperparameters=None):
    #     """Creates a neural network for the agents to use"""
    #     if hyperparameters is None: hyperparameters = self.hyperparameters
    #     if key_to_use: hyperparameters = hyperparameters[key_to_use]
    #     if override_seed:
    #         seed = override_seed
    #     else:
    #         seed = self.config.seed
    #
    #     default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
    #                                       "initialiser": "default", "batch_norm": False,
    #                                       "columns_of_data_to_be_embedded": [],
    #                                       "embedding_dimensions": [], "y_range": ()}
    #
    #     for key in default_hyperparameter_choices:
    #         if key not in hyperparameters.keys():
    #             hyperparameters[key] = default_hyperparameter_choices[key]
    #
    #     return NN(input_dim=input_dim, layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
    #               output_activation=hyperparameters["final_layer_activation"],
    #               batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
    #               hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
    #               columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
    #               embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"],
    #               random_seed=seed).to(self.device)

    def load_model(self, net, file_path, map_location):
        state_dict = torch.load(file_path, map_location=map_location)
        net.load_state_dict(state_dict)

    #
    # def create_replay_buffer(self, replay_buffer_file):
    #     if replay_buffer_file == "":
    #         memory = PriorityReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size,
    #                                       device=self.device,
    #                                       normalizer=None, seed=self.seed)
    #     else:
    #         memory = pickle.load(open(replay_buffer_file, 'rb'))
    #         print("loaded replay buffer size:{}".format(self.memory.size))
    #     return memory

    # if self.time_step % 100 == 0:
    #     print("save replay buffer to disk")
    #     f = open("buffer.obj", 'wb')
    #     pickle.dump(self.memory, f)

    # def store_model(self, file_path):
    #     torch.save(self.q_network_local.state_dict(), file_path)

    def reset(self):
        self.episode_number += 1

    # def pick_action(self, frame, robot_pose):
    #     """
    #
    #     :param frame: [w,h]
    #     :param robot_pose: [1,2]
    #     :return:
    #     """
    #     rnd = random.random()
    #     if self.params['is_train']:
    #         self.eps = max(self.eps * self.eps_decay, self.eps_min)
    #     else:
    #         self.eps = 0
    #     if rnd < self.eps:
    #         return np.random.randint(self.action_size)
    #     else:
    #         frame_in = torch.Tensor([frame]).to(self.device)
    #         robot_pose_in = torch.Tensor([robot_pose]).to(self.device)
    #
    #         self.policy_net.eval()
    #         with torch.no_grad():
    #             q_val = self.policy_net(frame_in, robot_pose_in)
    #
    #         action = np.argmax(q_val.cpu().data.numpy())
    #
    #     return action

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        # if not isinstance(network, list): network = [network]
        optimizer.zero_grad()  # reset gradients to 0
        loss.backward()  # this calculates the gradients
        self.logger.info("Loss -- {}".format(loss.item()))
        if self.debug_mode: self.log_gradient_and_weight_information(network, optimizer)
        if clipping_norm is not None:
            for p in network.parameters():
                p.grad.data.clamp_(-1, 1)
            # for net in network:
            #     torch.nn.utils.clip_grad_norm_(net.parameters(),
            #                                    clipping_norm)  # clip gradients to help stabilise training
        optimizer.step()  # this applies the gradients

    # def learn(self, memory_config_dir):
    #     self.policy_net.train()
    #     self.target_net.eval()
    #     loss_value = 0
    #
    #     # if len(self.memory) > self.batch_size:
    #     if True:
    #         tree_idx, minibatch, ISWeights = self.memory.sample()
    #         frames_in, robot_poses_in, actions, rewards, next_frames_in, next_robot_poses_in, dones = minibatch
    #
    #         # Get the action with max Q value
    #         if self.is_double:
    #             q_values = self.policy_net(next_robot_poses_in).detach()
    #             max_action_next = q_values.max(1)[1].unsqueeze(1)
    #             Q_target = self.target_net(next_frames_in, next_robot_poses_in).gather(1, max_action_next)
    #             Q_target = rewards + (self.gamma * Q_target * (1 - dones))
    #             Q_expected = self.policy_net(frames_in, robot_poses_in).gather(1, actions)
    #         else:
    #             q_values = self.target_net(next_frames_in, next_robot_poses_in).detach()
    #             max_action_values = q_values.max(1)[0].unsqueeze(1)
    #             # If done just use reward, else update Q_target with discounted action values
    #             Q_target = rewards + (self.gamma * max_action_values * (1 - dones))
    #             Q_action_values = self.policy_net(frames_in, robot_poses_in)
    #             Q_expected = Q_action_values.gather(1, actions)
    #             # self.update_q_action_values(Q_action_values, robot_poses_in)
    #
    #         self.optimizer.zero_grad()
    #         loss = self.weighted_mse_loss(Q_expected, Q_target, ISWeights)
    #         loss.backward()
    #
    #         for p in self.policy_net.parameters():
    #             p.grad.data.clamp_(-1, 1)
    #         self.optimizer.step()
    #         loss_value = loss.item()
    #
    #         Q_expected2 = self.policy_net(frames_in, robot_poses_in).gather(1, actions)
    #         loss_each_item = torch.abs(Q_expected2 - Q_target)
    #
    #         loss_reward_each_item = loss_each_item + rewards
    #         loss_reward_each_item = loss_reward_each_item.detach().cpu().numpy()
    #         tree_idx = tree_idx[:, np.newaxis]
    #         self.memory.batch_update(tree_idx, loss_reward_each_item)
    #     if self.time_step % self.update_every == 0:
    #         self.update_target_network()
    #     return loss_value

    # def weighted_mse_loss(self, input, target, weight):
    #     return torch.sum(torch.from_numpy(weight).to(self.device) * (input - target) ** 2)

    # def update_target_network(self):
    #     self.target_net.load_state_dict(self.policy_net.state_dict())

    def setup_logger(self):
        """Sets up the logger"""
        filename = "Training.log"
        try:
            if os.path.isfile(filename):
                os.remove(filename)
        except:
            pass

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # create a file handler
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(handler)
        return logger

    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.memory) > self.hyperparameters["batch_size"]

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def log_gradient_and_weight_information(self, network, optimizer):

        # log weight information
        total_norm = 0
        for name, param in network.named_parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.logger.info("Gradient Norm {}".format(total_norm))

        for g in optimizer.param_groups:
            learning_rate = g['lr']
            break
        self.logger.info("Learning Rate {}".format(learning_rate))

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
