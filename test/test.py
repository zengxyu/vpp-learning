import math
import pickle

import numpy as np


class Parent(object):
    def __init__(self):
        print("Parent init")

    def say(self):
        print("I am parent")
        self.test()

    def test(self):
        print("pppp")


class Children(Parent):
    def __init__(self):
        super(Children, self).__init__()
        print("Children init")

    def say(self):
        super(Children, self).say()

    def test(self):
        print("cccccc")


if __name__ == '__main__':
    # arr = np.array([0, 0, 0, 1, 1])
    # print(np.sum(arr == 1))
    # print(math.log2(3000))
    # a = np.array([[1, 2, -1, -1], [1, -1, -1, -1]])
    # a[a < 0] = 0
    # print(a)
    path = "/home/zeng/workspace/vpp-learning/output_remote6/out_p3d_original/experience/buffer.obj"
    path2 = "/home/zeng/workspace/vpp-learning/output_remote8/out_p3d_random_env_spacial_lstm/experience/buffer.obj"
    path3 = "/home/zeng/workspace/vpp-learning/output_remote8/out_ros_static_env_seq_len_10_2/experience/buffer.obj"
    path4 = "/home/zeng/workspace/vpp-learning/output_remote55/out_p3d_static_env_seq_len_10_spacial/experience/buffer.obj"
    file = open(path4, 'rb')
    memory = pickle.load(file)
    memory.sample()
    print()

    # print(math.log2(3000))
    # a = np.array([[1, 2, -1, -1], [1, -1, -1, -1]])
    # a[a < 0] = 0
    # print(a)
    a = 0.98 ** 100 * 1e-4
    print(a)
# if __name__ == '__main__':
#     child = Children()
#     child.say()
# import torch
#
# a = torch.rand(4, 3, 28, 28)
# ind = torch.tensor([0, 2])
# c = a.index_select(0, ind)
# print(c.shape)
