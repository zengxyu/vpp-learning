import numpy as np

print(np.e ** -0.93)
print(np.e ** -0.5)
print(np.e ** -0.25)
array = np.array([[1, 2], [2, 2], [3, 2]])
b = np.array([[3], [2], [2]])
print(array / b)


def nan_to_num(n):
    NEAR_0 = 1e-15
    return np.clip(n, NEAR_0, 1 - NEAR_0)


print(nan_to_num(np.NaN))
