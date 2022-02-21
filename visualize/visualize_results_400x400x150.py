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
from visualize.plot_format import init_plt, set_plt_format

font_size = 14
colors = ["#ff6666", "#6666ff", "#009999", "#ff9933", "#669900"]
font1 = {'size': font_size, }


def read_data(data_dir):
    path = os.path.join(data_dir, "result_log", "ZEvaluation_log.pkl")
    file = open(path, "rb")
    env_config = read_yaml(os.path.join(data_dir, "configs"), "env.yaml")
    plant_models_dir = os.path.join(get_project_path(), "data", 'plant_models')

    plants = load_plants(plant_models_dir, env_config["plant_types"], env_config["roi_neighbors"],
                         env_config["resolution"])
    observable_roi_total, observable_occ_total = count_observable_cells(env_config, plants)

    return pickle.load(file), observable_roi_total, observable_occ_total


def plot_coverage_by_time_step(data_dir, data, observable_occ_total):
    occupied_cells = np.array(data['new_occupied_cells'])
    rois_cells = np.array(data["new_found_rois"])
    occupied_rates = np.mean(occupied_cells, axis=0) / observable_occ_total
    rois_rates = np.mean(rois_cells, axis=0) / observable_roi_total

    num_time_step = rois_cells.shape[1]
    xs = np.linspace(0, num_time_step, num_time_step)

    ylims = [0.8, 0.7]
    rates = [occupied_rates, rois_rates]
    y_labels = ["Occupied cell coverage rate", "ROIs coverage rate"]
    names = ["p3d_occ_coverage_rate.png", "p3d_rois_coverage_rate.png"]

    for i, rate in enumerate(rates):
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
        plt.plot(xs, np.cumsum(rate), colors[i % len(colors)])

        plt.ylabel(y_labels[i])
        plt.xlabel("Time step")
        plt.ylim(0, ylims[i])

        # plt.legend()
        plt.savefig(os.path.join(data_dir, "result_log", names[i]))
        plt.clf()
        plt.close()


if __name__ == '__main__':
    data_dir = "evaluate_action10_sensor300_smallobs_maxsteps400"
    data_dir = os.path.join(get_project_path(), "output", data_dir)
    data, observable_roi_total, observable_occ_total = read_data(data_dir)
    plot_coverage_by_time_step(data_dir, data, observable_occ_total)
