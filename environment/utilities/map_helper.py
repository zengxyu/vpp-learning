#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 1/24/22 3:24 PM 
    @Description    :
        
===========================================
"""
import math as m
import numpy as np
from scipy.ndimage.filters import gaussian_filter, sobel


def count_neighbor_rate(unknown_map, known_free_map, known_target_map, neighbor_layer=2):
    unknown_map_nb = unknown_map[:neighbor_layer, :, :].copy()
    known_free_map_nb = known_free_map[:neighbor_layer, :, :].copy()
    known_target_map_nb = known_target_map[:neighbor_layer, :, :].copy()
    # make the item < 0 to be 0
    unknown_map_nb[unknown_map_nb < 0] = 0
    known_free_map_nb[known_free_map_nb < 0] = 0
    known_target_map_nb[known_target_map_nb < 0] = 0

    unknown_num = np.sum(unknown_map_nb)
    known_free_num = np.sum(known_free_map_nb)
    known_target_num = np.sum(known_target_map_nb)

    known_num = known_free_num + known_target_num
    total_num = known_num + unknown_num

    known_target_rate = known_target_num / (known_num + 1)
    unknown_rate = unknown_num / total_num

    return known_target_rate, unknown_rate


def transform_map(unknown_map, known_free_map, known_target_map):
    unknown_map = np.array(unknown_map)
    known_free_map = np.array(known_free_map)
    known_target_map = np.array(known_target_map)
    sum_map = unknown_map + known_free_map + known_target_map
    sum_map = np.sum(sum_map) + 1e-15
    unknown_map_prob = unknown_map / sum_map
    known_free_map_prob = known_free_map / sum_map
    known_target_map_prob = known_target_map / sum_map

    unknown_map_prob_f = sobel(gaussian_filter(unknown_map_prob, sigma=7))
    known_free_map_prob_f = sobel(gaussian_filter(known_free_map_prob, sigma=7))
    known_target_map_prob_f = sobel(gaussian_filter(known_target_map_prob, sigma=7))
    map = np.concatenate(
        [unknown_map_prob, known_free_map_prob, known_target_map_prob, unknown_map_prob_f, known_free_map_prob_f,
         known_target_map_prob_f], axis=0)
    return map

    # def compute_global_map(self):
    #     res = np.zeros(shape=(3, 32, 32, 32))
    #     for i in range(0, 256, 8):
    #         for j in range(0, 256, 8):
    #             for k in range(0, 256, 8):
    #                 res[0, i // 8, j // 8, k // 8] = np.sum(self.known_map[i:i + 8, j:j + 8, k:k + 8] == 0)
    #                 res[1, i // 8, j // 8, k // 8] = np.sum(self.known_map[i:i + 8, j:j + 8, k:k + 8] == 1)
    #                 res[2, i // 8, j // 8, k // 8] = np.sum(self.known_map[i:i + 8, j:j + 8, k:k + 8] == 2)
    #     return res


def cart2sph(x, y, z):
    XsqPlusYsq = x ** 2 + y ** 2
    r = m.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = m.atan2(z, m.sqrt(XsqPlusYsq))  # theta
    az = m.atan2(y, x)  # phi
    return az, elev, r


def robot_pose_cart_2_polor(pos):
    return cart2sph(pos[0], pos[1], pos[2])


def make_up_map(map):
    # 15 * 36 * 18
    # 5 * 90 * 45
    map = np.reshape(map, (3, 2, 2, 18, 2, 9))
    # 3 x 2 x 2 x 5 x 18 x 9
    map = np.transpose(map, (1, 2, 4, 0, 3, 5))

    # 5 * 4 * 2 * 15 * 15
    map = np.reshape(map, (8, 3, -1))

    map = np.sum(map, axis=2)

    denominator = np.sum(map, axis=1)[:, np.newaxis]
    map = map / denominator

    return map


def make_up_8x15x9x9_map(one_map):
    # 5 * 90 * 45
    one_map = np.reshape(one_map, (15, 4, 9, 2, 9))
    one_map = np.transpose(one_map, (1, 3, 0, 2, 4))
    # 5 * 4 * 2 * 15 * 15
    one_map = np.reshape(one_map, (8, 15, 9, 9))
    return one_map


def sum_block(one_map):
    one_map = np.reshape(one_map, (5, 36, 10, 18, 10))
    one_map = np.transpose(one_map, (0, 1, 3, 2, 4))
    one_map = np.reshape(one_map, (5, 36, 18, 100))
    one_map = np.sum(one_map, axis=-1)
    return one_map
