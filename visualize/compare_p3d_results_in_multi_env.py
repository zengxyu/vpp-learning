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
    plant_dict = {}
    for plant_type, plant in zip(env_config["plant_types"], all_plants):
        plant_dict[plant_type] = plant

    observable_roi_total_list = []
    observable_occ_total_list = []
    for plant_types in plant_types_list:
        plants = [all_plants[env_config["plant_types"].index(plant_type)] for plant_type in plant_types]
        observable_roi_total, observable_occ_total = count_observable_cells(env_config, plant_types, plants)
        observable_roi_total_list.append(observable_roi_total)
        observable_occ_total_list.append(observable_occ_total)

    return data, observable_roi_total_list, observable_occ_total_list


def compute_occ_roi_rates(data_dir):
    data, observable_roi_total, observable_occ_total = read_data(data_dir)
    occupied_cells = np.array(data['new_occupied_cells'])
    rois_cells = np.array(data["new_found_rois"])
    occupied_rate = np.mean(np.sum(occupied_cells[:, :300], axis=1)) / observable_occ_total
    rois_rate = np.mean(np.sum(rois_cells[:, :300], axis=1)) / observable_roi_total
    return occupied_rate, rois_rate


if __name__ == '__main__':
    evaluation_root = os.path.join(get_project_path(), "output", "evaluation2")
    evaluation_dirs = os.listdir(evaluation_root)
    data_dir_paths = [os.path.join(evaluation_root, evaluation_dir) for evaluation_dir in evaluation_dirs]

    for data_dir_path in data_dir_paths:
        occupied_rate, rois_rate = compute_occ_roi_rates(data_dir_path)
        print("{}: occupied_rate : {}; rois_rate : {}".format(data_dir_path, occupied_rate, rois_rate))
