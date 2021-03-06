#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/20/22 10:21 PM 
    @Description    :
        
===========================================
"""
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect

colors = ["#ff6666", "#6666ff", "#009999", "#ff9933", "#669900"]
# rgb_colors = [(255, 102, 102), (102, 102, 255), (0, 153, 153), (256, 153, 51), (102, 153, 0)]
rgb_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (102, 153, 0)]
# rgb_colors = [(255, 50, 50), (50, 255, 50), (50, 50, 255), (255, 255, 0), (102, 153, 0)]


def init_plt(i):
    fig = plt.figure(i, figsize=figaspect(1))
    return plt


def set_plt_format(plt):
    FONT_SIZE = 14

    plt.rc('font', size=FONT_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONT_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title

    plt.grid()
    plt.tight_layout()
