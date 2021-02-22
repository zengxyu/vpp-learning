import numpy as np


class ObservedMapRecord:
    def __init__(self, capacity=50000, observed_map_shape=(3, 18, 18), max_episode=1000):
        self.capacity = capacity
        self.max_episode = max_episode
        self.cursor = 0
        self.size = 0
        self.observed_maps = np.zeros(
            (self.capacity, observed_map_shape[0], observed_map_shape[1], observed_map_shape[2]))
        self.observed_map_mean = np.zeros(observed_map_shape)
        self.observed_map_std = np.zeros(observed_map_shape)

    def add_observed_map(self, observed_map):
        self.observed_maps[self.cursor] = observed_map
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if self.cursor % self.max_episode == 0:
            self.observed_map_mean = np.mean(self.observed_maps[:self.size], axis=0)
            self.observed_map_std = np.std(self.observed_maps[:self.size], axis=0)

    def get_observed_map_mean(self):
        return self.observed_map_mean

    def get_observed_map_std(self):
        return self.observed_map_std
