#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/21/22 2:06 AM 
    @Description    :
        
===========================================
"""
import os

import numpy as np

from utilities.util import get_project_path
from visualize.visualize_results_400x400x150 import read_data


def compute_occ_roi_rates(data_dir):
    data_dir = os.path.join(get_project_path(), "output", data_dir)
    data, observable_roi_total, observable_occ_total = read_data(data_dir)
    occupied_cells = np.array(data['new_occupied_cells'])
    rois_cells = np.array(data["new_found_rois"])
    occupied_rate = np.mean(np.sum(occupied_cells[:, :300], axis=1)) / observable_occ_total
    rois_rate = np.mean(np.sum(rois_cells[:, :300], axis=1)) / observable_roi_total
    return occupied_rate, rois_rate


if __name__ == '__main__':
    data_dirs = ["evaluate_action10_sensor300_smallobs_maxsteps400",
                 "evaluate_action10_sensor300_smallobs_maxsteps400_300x500x150",
                 "evaluate_action10_sensor300_smallobs_maxsteps400_500x500x150",
                 "evaluate_action10_sensor300_smallobs_maxsteps400_500x500x150_6plants",
                 "evaluate_action10_sensor300_smallobs_maxsteps400_500x500x150_8plants"

                 ]
    for data_dir in data_dirs:
        occupied_rate, rois_rate = compute_occ_roi_rates(data_dir)
        print("{}: occupied_rate : {}; rois_rate : {}".format(data_dir, occupied_rate, rois_rate))
