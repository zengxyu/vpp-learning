import numpy as np


class RewardRecord:
    def __init__(self, capacity=50000, max_episode=1000):
        self.capacity = capacity
        self.max_episode = max_episode
        self.cursor = 0
        self.size = 0
        self.rewards = np.zeros(self.capacity)
        self.reward_mean = 0
        self.reward_std = 0

    def add_reward(self, reward):
        self.rewards[self.cursor] = reward
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if self.cursor % self.max_episode == 0:
            self.reward_mean = np.mean(self.rewards[:self.size])
            self.reward_std = np.std(self.rewards[:self.size])

    def get_reward_mean(self):
        return self.reward_mean

    def get_reward_std(self):
        return self.reward_std


if __name__ == '__main__':
    reward_record = RewardRecord()
    for i in range(5):
        reward_record.add_reward(i)
        print()
        print(reward_record.rewards[:reward_record.size])
        print(reward_record.get_reward_mean())
