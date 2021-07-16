import copy
import random
import pickle
import os
import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt


class Trainer(object):
    """Runs games for given agents. Optionally will visualise and save the results"""

    def __init__(self, config, agent):
        self.config = config
        self.agent = agent

    def train_p3d(self):
        pass

    def train_gym(self):
        pass

    def print_two_empty_lines(self):
        print("-----------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------")
        print(" ")

    def save_obj(self, obj, name):
        """Saves given object as a pickle file"""
        if name[-4:] != ".pkl":
            name += ".pkl"
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        """Loads a pickle file object"""
        with open(name, 'rb') as f:
            return pickle.load(f)
