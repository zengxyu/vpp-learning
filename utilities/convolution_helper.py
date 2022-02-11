#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/11/22 11:21 AM 
    @Description    :
        
===========================================
"""


def compute_conv_out_width(i, k, s, p):
    """
    计算卷积输出层的宽度
    :param i: 输入尺寸
    :param k: 卷积核大小
    :param s: 步幅
    :param p: 边界扩充
    :return: 输出的feature map的宽
    """
    o = (i - k + 2 * p) / s + 1
    return int(o)


def compute_de_conv_out_width(i, k, s, p):
    """
    计算反卷积输出层的宽度
    :param i: 输入尺寸
    :param k: 卷积核大小
    :param s: 步幅
    :param p: 边界扩充
    :return: 输出的feature map的宽
    """
    out = (i - 1) * s + k - 2 * p
    return int(out)


def compute_conv_out_node_num(d, w, h):
    """
    计算卷积后输出层的节点数量
    :param d: depth channel number
    :param w: width
    :param h: height
    :return:
    """
    return int(d * w * h)


if __name__ == '__main__':
    # print(get_project_path())
    a = compute_conv_out_width(i=6, k=4, s=2, p=0)
    b = compute_conv_out_width(i=18, k=4, s=2, p=1)

    print(a, b)
    c = compute_conv_out_node_num(9, 18, 50)
    print(c)
