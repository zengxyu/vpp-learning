import random
import torch

from rl_agents.agents.Model_Helper import ModelHelper
from utils.basic_logger import BasicLogger


class BaseAgent(ModelHelper):
    def __init__(self, config, network_cls, action_space):
        self.name = "grid world"
        self.seed = random.seed(42)
        self.config = config
        self.debug_mode = config.debug_mode
        self.logger = BasicLogger.setup_console_logging(config)

        self.device = torch.device("cuda") if torch.cuda.is_available() and config.use_GPU else torch.device("cpu")
        print("device:", self.device)
        self.hyper_parameters = None

        self.memory = None

        self.average_score_required_to_win = config.reward_threshold
        self.turn_off_exploration = False
        self.global_step_number = 0
        self.episode_number = 0
        self.network_cls = network_cls
        self.action_space = action_space
        self.model_dict, self.optimizer_dict = self.__build_model_and_optimizer_dict()

    def __build_model_and_optimizer_dict(self):
        return None, None

    def store_model(self):
        print("Save model to path : {}".format(self.config.out_model))
        self.store_model_optimizer(self.model_dict, self.optimizer_dict, self.config.out_model, "Agent",
                                   self.episode_number)

    def load_model(self, index):
        print("Load model from path : {}".format(self.config.in_model))

        self.load_model_optimizer(self.model_dict, self.optimizer_dict, self.config.in_model, "Agent", index,
                                  self.device)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        # if not isinstance(network, list): network = [network]
        optimizer.zero_grad()  # reset gradients to 0
        loss.backward()  # this calculates the gradients
        self.logger.info("Loss -- {}".format(loss.item()))
        if self.debug_mode:
            self.log_gradient_and_weight_information(network, optimizer)
        if clipping_norm is not None:
            for p in network.parameters():
                p.grad.data.clamp_(-1, 1)
            # for net in network:
            #     torch.nn.utils.clip_grad_norm_(net.parameters(),
            #                                    clipping_norm)  # clip gradients to help stabilise training
        optimizer.step()

    def step(self, state, action, reward, next_state, done):
        self.memory.add_experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.global_step_number += 1
        if done:
            self.episode_number += 1
            # update learning rate

    def time_for_q_network_to_learn(self):
        return self.enough_experiences_to_learn_from() and self.right_amount_replay_start_size()

    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.memory) > self.hyper_parameters["batch_size"]

    def right_amount_replay_start_size(self):
        """if the replay buffer's size is less than replay_start_size, skip update"""
        return len(self.memory) > self.hyper_parameters["replay_start_size"]

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

    def update_learning_rate(self, starting_lr, optimizer, rolling_reward):
        """Lowers the learning rate according to how close we are to the solution"""
        if rolling_reward > 0.75 * self.average_score_required_to_win:
            new_lr = starting_lr / 100.0
        elif rolling_reward > 0.6 * self.average_score_required_to_win:
            new_lr = starting_lr / 20.0
        elif rolling_reward > 0.5 * self.average_score_required_to_win:
            new_lr = starting_lr / 10.0
        elif rolling_reward > 0.25 * self.average_score_required_to_win:
            new_lr = starting_lr / 2.0
        else:
            new_lr = starting_lr
        for g in optimizer.param_groups:
            g['lr'] = new_lr
        if random.random() < 0.001: self.logger.info("Learning rate {}".format(new_lr))

    def print_summary_of_latest_evaluation_episode(self):
        """Prints a summary of the latest episode"""
        print(" ")
        print("----------------------------")
        print("Episode score {} ".format(self.total_episode_score_so_far))
        print("----------------------------")
