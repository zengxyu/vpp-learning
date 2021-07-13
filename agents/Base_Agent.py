import logging
import os
import sys
import gym
import random
import numpy as np
import torch
import time
# import tensorflow as tf
# from tensorboardX import SummaryWriter
from torch.optim import optimizer


class Base_Agent(object):
    def __init__(self, config):
        self.name = "grid world"
        self.seed = random.seed(42)
        self.logger = self.setup_logger()
        self.config = config
        self.debug_mode = config.debug_mode

        self.device = torch.device("cuda") if torch.cuda.is_available() and config.use_GPU else torch.device("cpu")
        print("device:", self.device)

        self.hyperparameters = None

        self.memory = None

        self.average_score_required_to_win = config.environment['reward_threshold']
        self.turn_off_exploration = False
        self.global_step_number = 0
        self.episode_number = 0

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None):
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
        optimizer.step()

    def reset(self, rolling_reward):
        self.episode_number += 1

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

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
