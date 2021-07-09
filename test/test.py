# import numpy as np
#
# z = np.zeros((2, 2, 6), dtype='U2')
# o = np.ones((2, 1), dtype='O')
# print("z\n", z)
# print("o\n", o)
# c = np.hstack([[o,z], z])
# print(c)
# def calculate_epsilon_with_inverse_strategy(epsilon, episode_number, epsilon_decay_denominator):
#     """Calculates epsilon according to an inverse of episode_number strategy"""
#     epsilon = epsilon / (1.0 + (episode_number / epsilon_decay_denominator))
#     return epsilon
#
#
# epsilon = 1.0
# episode_number = 12
# print(calculate_epsilon_with_inverse_strategy(epsilon, episode_number=12, epsilon_decay_denominator=1))
# print(calculate_epsilon_with_inverse_strategy(epsilon, episode_number=20, epsilon_decay_denominator=1))
# print(0.5*0.999**500)
# import pickle
# import os
# import torch
#
# from memory.replay_buffer import PriorityReplayBuffer
#
# exp_in_path = os.path.join("../output_reset_and_random3", "experience", "buffer.obj")
# buffer_size = 4000
# memory1 = PriorityReplayBuffer(buffer_size=buffer_size, batch_size=128,
#                                device=torch.device("cpu"),
#                                normalizer=None, seed=40)
#
# memory2 = pickle.load(open(exp_in_path, 'rb'))
# vs, data = memory2.get_all_experiences()
# memory2.tree
# tree[tree_idx] = data
#
# memory1.preload_experiences([vs[:buffer_size], data[:buffer_size]])
# print()
import time

import numpy as np

a = [[0, 1], [1, 0], [2, 3], [2, 4], [3, 5]]
time0 = time.time()
b = np.vstack([i for i in a])  # 0.000063
time1 = time.time()

c = np.concatenate([i for i in a], axis=0)  # 0.000040
time2 = time.time()

d = np.conjugate([i for i in a])
time3 = time.time()

print(b, time1 - time0)
print(c, time2 - time1)
print(d, time3 - time2)
