#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/20/22 9:24 PM 
    @Description    :
        
===========================================
"""
import os.path
import pickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect

from config import read_yaml
from environment.utilities.count_cells_helper import count_observable_cells
from environment.utilities.plant_models_loader import load_plants
from utilities.util import get_project_path
from visualize.compare_p3d_results_in_multi_env import read_data
from visualize.plot_format import init_plt, set_plt_format

font_size = 14
colors = ["#ff6666", "#6666ff", "#009999", "#ff9933", "#669900"]
font1 = {'size': font_size, }


def get_rates_mean_std(occupied_cells_episodes, observable_occ_episodes):
    occupied_cells_episodes = np.cumsum(occupied_cells_episodes, axis=1)
    occupied_rates_episodes = []
    for i, occupied_cells in enumerate(occupied_cells_episodes):
        occupied_rates_episodes.append(occupied_cells / observable_occ_episodes[i])
    occupied_rates_episodes = np.array(occupied_rates_episodes)
    occupied_rates_mean = np.mean(occupied_rates_episodes, axis=0)
    occupied_rates_std = np.std(occupied_rates_episodes, axis=0)
    return occupied_rates_mean, occupied_rates_std


def plot_coverage_by_time_step(data_dir, data, observable_roi_totals, observable_occ_totals):
    rois_cells_episodes = np.array(data["new_found_rois"])
    occupied_cells_episodes = np.array(data['new_occupied_cells'])
    rois_rates_mean, rois_rates_std = get_rates_mean_std(rois_cells_episodes, observable_roi_totals)
    occupied_rates_mean, occupied_rates_std = get_rates_mean_std(occupied_cells_episodes, observable_occ_totals)

    num_time_step = rois_cells_episodes.shape[1]
    xs = np.linspace(0, num_time_step, num_time_step)

    ylims = [1, 1]
    rates = [occupied_rates_mean, rois_rates_mean]
    stds = [occupied_rates_std, rois_rates_std]
    y_labels = ["Occupied cell coverage rate", "ROIs coverage rate"]
    names = ["p3d_occ_coverage_rate.png", "p3d_rois_coverage_rate.png"]
    i = 0
    for rate, std in zip(rates, stds):
        fig = plt.figure(i, figsize=figaspect(1))
        FONT_SIZE = 15

        plt.rc('font', size=FONT_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=FONT_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=FONT_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=FONT_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=FONT_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=FONT_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title

        plt.grid()
        plt.tight_layout()
        plt.fill_between(
            xs,
            rate - std,
            rate + std,
            color="b",
            alpha=0.1
        )
        plt.plot(xs, rate, 'b')

        plt.ylabel(y_labels[i])
        plt.xlabel("Time step")
        plt.ylim(0, ylims[i])

        # plt.legend()
        plt.savefig(os.path.join(data_dir, "result_log", names[i]))
        plt.clf()
        plt.close()
        i += 1


if __name__ == '__main__':
    data_dir = "evaluation/320x320x150_2plants"
    data_dir = os.path.join(get_project_path(), "output", data_dir)
    data, observable_roi_total, observable_occ_total = read_data(data_dir)
    plot_coverage_by_time_step(data_dir, data, observable_roi_total, observable_occ_total)
