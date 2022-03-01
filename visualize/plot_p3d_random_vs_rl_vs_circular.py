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
import numpy as np
from utilities.util import get_project_path
from visualize.appearance_define import init_plt, set_plt_format
from visualize.compare_p3d_results_in_multi_env import read_data

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


def plot_coverage_by_time_step(data_dir, num_time_step, data):
    xs = np.linspace(0, num_time_step, num_time_step)

    ylims = [1, 1]
    y_labels = ["Coverage rate of ROI cells", "Coverage rate of Occupied cells"]
    names = ["p3d_rois_coverage_rate.png", "p3d_occ_coverage_rate.png"]
    colors = ['b', 'g', 'r']
    policy_labels = ["Random policy", "Perimeter policy", "Our RL-policy"]
    i = 0

    for item, label, name in zip(data, y_labels, names):
        plt = init_plt(i)
        set_plt_format(plt)
        count = 0
        for [policy_rate, policy_std], policy in zip(item, policy_labels):
            plt.fill_between(
                xs,
                policy_rate - policy_std,
                policy_rate + policy_std,
                color=colors[count],
                alpha=0.08
            )
            plt.plot(xs, policy_rate, colors[count], label=policy)
            count += 1
        plt.legend()
        plt.ylabel(y_labels[i])
        plt.xlabel("Time step")
        plt.ylim(0, ylims[i])

        # plt.legend()
        plt.savefig(os.path.join(data_dir, names[i]), bbox_inches='tight')
        plt.clf()
        plt.close()
        i += 1


def extract_rates_mean_std(data_dir_random_policy):
    data, roi_rates, occ_rates, observable_roi_totals, observable_occ_totals = read_data(data_dir_random_policy)

    rois_rates_mean, rois_rates_std = get_rates_mean_std(np.array(data["new_found_rois"]), observable_roi_totals)
    occupied_rates_mean, occupied_rates_std = get_rates_mean_std(np.array(data['new_occupied_cells']),
                                                                 observable_occ_totals)
    return rois_rates_mean, rois_rates_std, occupied_rates_mean, occupied_rates_std


if __name__ == '__main__':
    data_dir_random_policy = "evaluate_random_policy/400x400x150_4plants"
    data_dir_random_policy = os.path.join(get_project_path(), "output", data_dir_random_policy)
    rois_rates_mean_random, rois_rates_std_random, occupied_rates_mean_random, occupied_rates_std_random = extract_rates_mean_std(
        data_dir_random_policy)

    data_dir_rl_policy = "evaluation_p3d_0.15/400x400x150_4plants"
    data_dir_rl_policy = os.path.join(get_project_path(), "output", data_dir_rl_policy)
    rois_rates_mean_rl, rois_rates_std_rl, occupied_rates_mean_rl, occupied_rates_std_rl = extract_rates_mean_std(
        data_dir_rl_policy)

    data_dir_circular_policy = "evaluate_circular_policy/400x400x150_4plants"
    data_dir_circular_policy = os.path.join(get_project_path(), "output", data_dir_circular_policy)
    rois_rates_mean_circular, rois_rates_std_circular, occupied_rates_mean_circular, occupied_rates_std_circular = extract_rates_mean_std(
        data_dir_circular_policy)

    output_rl_vs_random = os.path.join(get_project_path(), "output", "compare_rl_circular_random")
    if not os.path.exists(output_rl_vs_random):
        os.makedirs(output_rl_vs_random)
    plot_coverage_by_time_step(output_rl_vs_random, len(rois_rates_mean_random),
                                   [[[rois_rates_mean_random, rois_rates_std_random],
                                     [rois_rates_mean_circular, rois_rates_std_circular],
                                     [rois_rates_mean_rl, rois_rates_std_rl]
                                     ],
                                    [[occupied_rates_mean_random, occupied_rates_std_random],
                                     [occupied_rates_mean_circular, occupied_rates_std_circular],
                                     [occupied_rates_mean_rl, occupied_rates_std_rl]]
                                    ])
