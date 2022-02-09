#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/9/22 7:32 PM 
    @Description    :
        
===========================================
"""
import glob
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "capnp"))
import capnp
import voxelgrid_capnp


class TYPE:
    ROI_NEIGHBORS = "roi_neighbors"
    NO_ROI_NEIGHBORS = "no_roi_neighbors"


def get_plant_path(plant_dir: str, plant_type: str, roi_neighbors: bool, resolution: float):
    """
    load plant from data/plant_models
    """
    if roi_neighbors:
        path_format = os.path.join(plant_dir, plant_type, "roi_neighbors", "*{}*.cvx".format(resolution))
    else:
        path_format = os.path.join(plant_dir, plant_type, "no_roi_neighbors", "*{}*.cvx".format(resolution))
    path = glob.glob(path_format)[0]
    return path


def read_from_local(file_path):
    """
    read from local file
    """
    with open(file_path) as file:
        voxelgrid = voxelgrid_capnp.Voxelgrid.read(file, traversal_limit_in_words=2 ** 32)
    labels = np.asarray(voxelgrid.labels)
    shape = tuple(voxelgrid.shape)
    data = labels.reshape(shape)
    return data


def load_plants(plant_dir, plant_types=["VG07_6", "VG07_6_more_occ", "VG07_6_no_fruits", "VG07_6_one_fruit"],
                roi_neighbors=True, resolution=0.01):
    """
    load plants from plant dir by plant_types and roi_neighbors
    """
    plants = []
    for plant_type in plant_types:
        path = get_plant_path(plant_dir, plant_type, roi_neighbors, resolution)
        plant = read_from_local(path)
        plants.append(plant)
    return plants
