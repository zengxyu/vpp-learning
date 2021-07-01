# import numpy as np
#
# z = np.zeros((2, 2, 6), dtype='U2')
# o = np.ones((2, 1), dtype='O')
# print("z\n", z)
# print("o\n", o)
# c = np.hstack([[o,z], z])
# print(c)
def calculate_epsilon_with_inverse_strategy(epsilon, episode_number, epsilon_decay_denominator):
    """Calculates epsilon according to an inverse of episode_number strategy"""
    epsilon = epsilon / (1.0 + (episode_number / epsilon_decay_denominator))
    return epsilon


epsilon = 1.0
episode_number = 12
print(calculate_epsilon_with_inverse_strategy(epsilon, episode_number=12, epsilon_decay_denominator=1))
print(calculate_epsilon_with_inverse_strategy(epsilon, episode_number=20, epsilon_decay_denominator=1))
