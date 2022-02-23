#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/18/22 3:13 PM 
    @Description    :
        
===========================================
"""
import numpy as np


def count_observable_cells(env_config, plant_types, plants):
    """
    calculate observable roi cells, observable occ cells
    """
    plant_observable_roi_ratios = env_config["plant_observable_roi_ratios"]
    plant_observable_occ_ratios = env_config["plant_observable_occ_ratios"]
    observable_roi_total = 0
    observable_occ_total = 0

    for plant, type in zip(plants, plant_types):
        observable_roi_total += np.sum(plant == 2) * plant_observable_roi_ratios[type]
        observable_occ_total += np.sum(plant == 1) * plant_observable_occ_ratios[type]
    return observable_roi_total, observable_occ_total


def count_cells(global_map):
    """
    calculate the number of roi cells, occ cells and free cells
    """
    roi_total = np.sum(global_map == 3)
    occ_total = np.sum(global_map == 2)
    free_total = np.sum(global_map == 1)
    return roi_total, occ_total, free_total
