import random
import numpy as np
import torch


class MemoryOptimizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, history_length, device, seed):
        # the capacity
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.history_length = history_length
        self.device = device
        random.seed(seed)

        # current data size
        self.count = 0
        # current cursor
        self.current = 0

        # data
        self.actions = np.empty((self.buffer_size, 1), dtype=np.int8)
        self.rewards = np.empty((self.buffer_size, 1), dtype=np.float)
        self.dones = np.empty((self.buffer_size, 1), dtype=np.bool)

        self.states = []
        self.pre_states = []
        self.post_states = []

    def add_experience(self, state, action, reward, next_state, done):
        # if self.states is initialized, initialize self.states
        if not self.states:
            # 2 * batch_size * history_length * obs_shape
            self.states = [None] * len(next_state)
            self.pre_states = [None] * len(next_state)
            self.post_states = [None] * len(next_state)

            for i in range(len(next_state)):
                next_obs = np.array(next_state[i])
                self.states[i] = np.empty((self.buffer_size, *np.shape(next_obs)), dtype=next_obs.dtype)
                self.pre_states[i] = np.empty((self.batch_size, self.history_length) + np.shape(next_obs),
                                              dtype=next_obs.dtype)
                self.post_states[i] = np.empty((self.batch_size, self.history_length) + np.shape(next_obs),
                                               dtype=next_obs.dtype)

        # add the data to the state buffer
        for i in range(len(next_state)):
            next_obs = np.array(next_state[i])
            assert np.shape(next_obs) == np.shape(self.states[i])[1:], \
                "Assert the observation {}, {}=={}".format(i, np.shape(next_obs), np.shape(self.states[i]))
            self.states[i][self.current, ...] = next_obs

        self.actions[self.current, ...] = action
        self.rewards[self.current, ...] = reward
        self.dones[self.current, ...] = done

        # if buffer is full, the count equal to self.buffer_size
        # if buffer is not full, the count equal to self.count + 1
        self.count = min(self.count + 1, self.buffer_size)
        self.current = (self.current + 1) % self.buffer_size

    def get_state(self, index):
        state = []
        for i in range(len(self.states)):
            obs = self.states[i][(index - (self.history_length - 1)):(index + 1), ...]
            state.append(obs)
        return state

    def sample(self):
        assert self.count > self.history_length
        indexes = []
        while len(indexes) < self.batch_size:
            while True:
                # sample an index
                index = random.randint(self.history_length, self.count - 1)
                # if wraps over current cursor, then get a new one:
                # because when the buffer is full, the cursor may be back and start from the first one
                if index - self.history_length < self.current <= index:
                    continue

                # if wraps over episode ned, then get new one
                if self.dones[(index - self.history_length):index].any():
                    continue
                break

            # having
            for i in range(len(self.pre_states)):
                self.pre_states[i][len(indexes), ...] = self.get_state(index - 1)[i]
                self.post_states[i][len(indexes), ...] = self.get_state(index)[i]
            indexes.append(index)

        # convert numpy to Tensor
        pre_states = [None] * len(self.states)
        post_states = [None] * len(self.states)
        for i in range(len(self.states)):
            pre_states[i] = torch.from_numpy(self.pre_states[i]).float().to(self.device)
            post_states[i] = torch.from_numpy(self.post_states[i]).float().to(self.device)
        actions = torch.from_numpy(self.actions[indexes]).int().to(self.device)
        rewards = torch.from_numpy(self.rewards[indexes]).float().to(self.device)
        dones = torch.from_numpy(self.dones[indexes]).int().to(self.device)

        if self.history_length == 1:
            for i in range(len(self.states)):
                pre_states[i] = torch.squeeze(pre_states[i], dim=1)
                post_states[i] = torch.squeeze(post_states[i], dim=1)
        return pre_states, actions, rewards, post_states, dones

    def __len__(self):
        return self.count
