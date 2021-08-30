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
import pickle

#
# path = "/home/zeng/workspace/vpp-learning/output/out_p3d_temporal_pose_random_108_control2/loss_reward/loss_reward.obj"
# f = open(path, "rb")
# a = pickle.load(f)

a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
b = np.reshape(a, (2, 2, 2))
c = np.transpose(b, (0, 2, 1))
print()