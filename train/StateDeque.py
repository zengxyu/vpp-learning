from collections import deque
import numpy as np


class State_DEQUE:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states_deque = deque([])
        self.next_states_deque = deque([])
        self.robot_poses = deque([])
        self.next_robot_poses = deque([])

    def __len__(self):
        assert self.states_deque.__len__() == self.next_states_deque.__len__()
        assert self.robot_poses.__len__() == self.next_robot_poses.__len__()
        assert self.states_deque.__len__() == self.robot_poses.__len__()
        return self.states_deque.__len__()

    def append(self, state, robot_pose):
        self.states_deque.append(state)
        self.robot_poses.append(robot_pose)
        if self.states_deque.__len__() > self.capacity:
            self.states_deque.popleft()
            self.robot_poses.popleft()

    def append_next(self, next_state, next_robot_pose):
        self.next_states_deque.append(next_state)
        self.next_robot_poses.append(next_robot_pose)
        if self.next_states_deque.__len__() > self.capacity:
            self.next_states_deque.popleft()
            self.next_robot_poses.popleft()

    def get_states(self):
        states = np.array(self.states_deque, dtype=float)
        return states

    def get_next_states(self):
        next_states = np.array(self.next_states_deque, dtype=float)
        return next_states

    def get_robot_poses(self):
        robot_poses = np.array(self.robot_poses, dtype=float)
        return robot_poses

    def get_next_robot_poses(self):
        next_robot_poses = np.array(self.next_robot_poses, dtype=float)
        return next_robot_poses

    def is_full(self):
        return self.__len__() == self.capacity


class Pose_State_DEQUE:
    def __init__(self, capacity):
        self.capacity = capacity
        self.robot_poses = deque([])
        self.next_robot_poses = deque([])

    def __len__(self):
        assert self.robot_poses.__len__() == self.next_robot_poses.__len__()
        return self.robot_poses.__len__()

    def append(self, robot_pose):
        self.robot_poses.append(robot_pose)
        if self.robot_poses.__len__() > self.capacity:
            self.robot_poses.popleft()

    def append_next(self, next_robot_pose):
        self.next_robot_poses.append(next_robot_pose)
        if self.next_robot_poses.__len__() > self.capacity:
            self.next_robot_poses.popleft()

    def get_robot_poses(self):
        robot_poses = np.array(self.robot_poses, dtype=float)
        return robot_poses

    def get_next_robot_poses(self):
        next_robot_poses = np.array(self.next_robot_poses, dtype=float)
        return next_robot_poses

    def is_full(self):
        return self.__len__() == self.capacity
