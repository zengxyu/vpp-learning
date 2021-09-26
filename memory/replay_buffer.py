import random
from collections import deque, namedtuple
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device, seed):
        """
        Replay memory allow old_agent to record experiences and learn from them

        Parametes
        ---------
        buffer_size (int): maximum size of internal memory
        batch_size (int): sample size from experience
        seed (int): random seed
        """
        self.device = device
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add_experience(self, state, action, reward, next_state, done):
        """Add experience"""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """
        Sample randomly and return (state, action, reward, next_state, done) tuple as torch tensors
        """
        "do normalization"
        num = self.__len__() if self.__len__() < self.batch_size else self.batch_size
        experiences = random.sample(self.memory, k=num)

        # Convert to torch tensors
        actions = torch.from_numpy(
            np.vstack([experience.action for experience in experiences if experience is not None])).float().to(
            self.device)
        rewards = torch.from_numpy(
            np.vstack([experience.reward for experience in experiences if experience is not None])).float().to(
            self.device)
        # Convert done from boolean to int
        dones = torch.from_numpy(
            np.vstack([experience.done for experience in experiences if experience is not None]).astype(
                np.uint8)).float().to(self.device)

        if isinstance(experiences[0].state, list):
            frames = torch.from_numpy(
                np.conjugate([experience.state[0] for experience in experiences if experience is not None])).float().to(
                self.device)
            robot_poses = torch.from_numpy(
                np.conjugate([experience.state[1] for experience in experiences if experience is not None])).float().to(
                self.device)
            frames_next = torch.from_numpy(
                np.conjugate(
                    [experience.next_state[0] for experience in experiences if experience is not None])).float().to(
                self.device)
            robot_poses_next = torch.from_numpy(
                np.conjugate(
                    [experience.next_state[1] for experience in experiences if experience is not None])).float().to(
                self.device)
            states = [frames, robot_poses]
            next_states = [frames_next, robot_poses_next]
        else:
            states = torch.from_numpy(
                np.conjugate([experience.state for experience in experiences if experience is not None])).float().to(
                self.device)
            next_states = torch.from_numpy(
                np.conjugate(
                    [experience.next_state for experience in experiences if experience is not None])).float().to(
                self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
