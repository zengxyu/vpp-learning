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
import pickle

import numpy as np

from config import read_yaml
from environment.utilities.count_cells_helper import count_observable_cells
from environment.utilities.plant_models_loader import load_plants
from utilities.util import get_project_path


def read_data(data_dir):
    path = os.path.join(data_dir, "result_log", "ZEvaluation_log.pkl")
    file = open(path, "rb")
    data, plant_types_list = pickle.load(file)

    env_config = read_yaml(os.path.join(data_dir, "configs"), "env.yaml")
    plant_models_dir = os.path.join(get_project_path(), "data", 'plant_models')

    all_plants = load_plants(plant_models_dir, env_config["plant_types"], env_config["roi_neighbors"],
                             env_config["resolution"])
    roi_rates = []
    occ_rates = []
    for i, plant_types in enumerate(plant_types_list):
        plants = [all_plants[env_config["plant_types"].index(plant_type)] for plant_type in plant_types]
        observable_roi_total, observable_occ_total = count_observable_cells(env_config, plant_types, plants)
        roi_sum = np.sum(data['new_found_rois'][i][:300]) + 1e-08
        occ_sum = np.sum(data['new_occupied_cells'][i][:300]) + 1e-08

        roi_rate = roi_sum / observable_roi_total
        occ_rate = occ_sum / observable_occ_total
        roi_rates.append(roi_rate)
        occ_rates.append(occ_rate)
        # print("i:{};plant_types:{};observable_roi_total:{};"
        #       "observable_occ_total:{}; rois sum:{}; occ sum:{}".format(i, plant_types, observable_roi_total,
        #                                                                 observable_occ_total, roi_sum, occ_sum))

    return data, np.array(roi_rates), np.array(occ_rates)


def compute_occ_roi_rates(data_dir):
    data, roi_rates, occ_rates = read_data(data_dir)
    occ_rates = np.mean(occ_rates)
    roi_rates = np.mean(roi_rates)

    return occ_rates, roi_rates


if __name__ == '__main__':
    evaluation_root = os.path.join(get_project_path(), "output", "evaluate_random_policy")
    evaluation_dirs = os.listdir(evaluation_root)
    evaluation_dirs = sorted(evaluation_dirs)
    data_dir_paths = [os.path.join(evaluation_root, evaluation_dir) for evaluation_dir in evaluation_dirs]

    for data_dir_path in data_dir_paths:
        occupied_rate, rois_rate = compute_occ_roi_rates(data_dir_path)
        print("{}:\n occupied_rate : {};\n rois_rate : {}".format(data_dir_path, occupied_rate, rois_rate))
