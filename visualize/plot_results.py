#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/15/22 3:31 PM 
    @Description    :
        
===========================================
"""
import argparse
import os.path
import pickle

import numpy as np

from utilities.util import get_project_path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--save_to", type=str, default="test_folder")
parser.add_argument("--in_folder", type=str, default=None)

parser_args = parser.parse_args()
parser_args.in_folder = os.path.join(get_project_path(), "output", parser_args.in_folder)
file = open(os.path.join("output", parser_args.in_folder, "result_log", "Train_log.pkl"), 'rb')
data = pickle.load(file)

font_size = 14
colors = ["#ff6666", "#6666ff", "#009999", "#ff9933", "#669900"]
font1 = {'size': font_size, }
font2 = {'size': font_size + 2}

keys = ['new_free_cells_rate', 'new_occ_cells_rate', 'new_roi_cells_rate']


def plot(data, save_path):
    count = 0
    for key, values in data.items():
        if key in keys:
            count += 1
            length = len(values)
            xs = np.arange(0, length, 1)
            ys = values
            plt.plot(xs, ys, colors[count % len(colors)], label=key)
            plt.xlabel("x")
            plt.ylabel("y")
    # plt.ylim(0, 1)
    plt.legend(prop=font1)
    plt.savefig(save_path)
    # plt.show()


if __name__ == '__main__':
    plot(data, save_path=os.path.join(parser_args.in_folder, "result.png"))
