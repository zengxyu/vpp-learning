import random

import numpy as np


class SumTree:
    """
    A binary tree data structure where the value of a parent is the sum of its children

    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Store the data with its priority in tree and data frameworks.

    Taken from: https://github.com/txzhao/rl-zoo/blob/master/DQN/priorExpReplay.py
    """

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.data_pointer = 0
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------parent nodes-------------][-------leaves to record priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    # [--------------data frame-------------]
    #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx] or self.tree[cr_idx] == 0.0:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx, self.data[data_idx]

    def get_leaf_random(self):
        leaf_idx = random.randint(self.capacity - 1, 2 * self.capacity - 2)
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class DoubleSumTree:
    """
    A binary tree data structure where the value of a parent is the sum of its children

    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Store the data with its priority in tree and data frameworks.

    Taken from: https://github.com/txzhao/rl-zoo/blob/master/DQN/priorExpReplay.py
    """

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.data_pointer = 0
        self.tree_unknown = np.zeros(2 * capacity - 1)
        self.tree_known = np.zeros(2 * capacity - 1)
        # [--------------parent nodes-------------][-------leaves to record priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    # [--------------data frame-------------]
    #             size: capacity
    def add(self, p_unknown, p_known, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update_unknown(tree_idx, p_unknown)  # update tree_frame
        self.update_known(tree_idx, p_known)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update_unknown(self, tree_idx, p):
        change = p - self.tree_unknown[tree_idx]
        self.tree_unknown[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree_unknown[tree_idx] += change

    def update_known(self, tree_idx, p):
        change = p - self.tree_known[tree_idx]
        self.tree_known[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree_known[tree_idx] += change

    def get_unknown_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree_unknown):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree_unknown[cl_idx] or self.tree_unknown[cr_idx] == 0.0:
                    parent_idx = cl_idx
                else:
                    v -= self.tree_unknown[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree_unknown[leaf_idx], data_idx, self.data[data_idx]

    def get_known_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree_known):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree_known[cl_idx] or self.tree_known[cr_idx] == 0.0:
                    parent_idx = cl_idx
                else:
                    v -= self.tree_known[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree_known[leaf_idx], data_idx, self.data[data_idx]

    def get_leaf_random(self):
        leaf_idx = random.randint(self.capacity - 1, 2 * self.capacity - 2)
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree_known[leaf_idx], self.data[data_idx]

    @property
    def total_p_unknown(self):
        return self.tree_unknown[0]  # the root

    @property
    def total_p_known(self):
        return self.tree_known[0]  # the root
