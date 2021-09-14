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
    # path = "/home/zeng/workspace/vpp-learning/output_remote6/out_p3d_original/experience/buffer.obj"
    # path2 = "/home/zeng/workspace/vpp-learning/output_remote8/out_p3d_random_env_spacial_lstm/experience/buffer.obj"
    # path3 = "/home/zeng/workspace/vpp-learning/output_remote8/out_ros_static_env_seq_len_10_2/experience/buffer.obj"
    # path4 = "/Users/weixianshi/PycharmProjects/vpp-learning/output/out_p3d_random_step_len_5_36_action_predict2/loss_reward/loss_reward.obj"

    path5 = "/Users/weixianshi/PycharmProjects/vpp-learning/output_evaluation/out_p3d_random_step_len_5_36_action_predict_440/loss_reward/loss_reward.obj"
    path6 = "/Users/weixianshi/PycharmProjects/vpp-learning/output_evaluation/out_p3d_random_step_len_5_36_action_predict_keep_training_model_440/loss_reward/loss_reward.obj"
    path7 = "/Users/weixianshi/PycharmProjects/vpp-learning/output_evaluation/out_p3d_random_to_static_step_len_5_36_action_predict_440/loss_reward/loss_reward.obj"
    path8 = "/Users/weixianshi/PycharmProjects/vpp-learning/output_evaluation/out_p3d_random_to_static_step_len_5_36_action_predict_keep_training_model_440/loss_reward/loss_reward.obj"
    total_num = 75370
    loss, rewards_without_training1 = pickle.load(open(path5, 'rb'))
    print("rewards_without_training1 avg:", np.mean(rewards_without_training1)/total_num)
    _, rewards_training2 = pickle.load(open(path6, 'rb'))
    print("rewards_training2 avg:", np.mean(rewards_training2)/total_num)

    _, rewards_static_env_without_training1 = pickle.load(open(path7, 'rb'))
    print("rewards_static_env_without_training1 avg:", np.mean(rewards_static_env_without_training1)/total_num)
    _, rewards_static_env_training2 = pickle.load(open(path8, 'rb'))
    print("rewards_static_env_training2 avg:", np.mean(rewards_static_env_training2)/total_num)

    """
    rewards_without_training1 avg: 38159.35
    rewards_training2 avg: 39929.8
    rewards_static_env_without_training1 avg: 29390.6
    rewards_static_env_training2 avg: 34111.55
    
    rewards_without_training1 avg: 0.5062936181504577
    rewards_training2 avg: 0.5297837335810004
    rewards_static_env_without_training1 avg: 0.3899509088496749
    rewards_static_env_training2 avg: 0.4525878996948388
    """
    # print(math.log2(3000))
    # a = np.array([[1, 2, -1, -1], [1, -1, -1, -1]])
    # a[a < 0] = 0
    # print(a)
    # a = 0.98 ** 100 * 1e-4
    # print(a)
# if __name__ == '__main__':
#     child = Children()
#     child.say()
# import torch
#
# a = torch.rand(4, 3, 28, 28)
# ind = torch.tensor([0, 2])
# c = a.index_select(0, ind)
# print(c.shape)
