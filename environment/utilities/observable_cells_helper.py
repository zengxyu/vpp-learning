#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/15/22 4:53 PM 
    @Description    :
        
===========================================
"""
import time

import numpy as np
import field_env_3d_helper
from field_env_3d_helper import Vec3D

left_surface_index = np.array([0, 4, 3, 7])
right_surface_index = np.array([1, 5, 2, 6])
front_surface_index = np.array([0, 4, 1, 5])
back_surface_index = np.array([3, 7, 2, 6])
upper_surface_index = np.array([4, 5, 7, 6])


def generate_vec3d_from_arr(arr):
    return Vec3D(*tuple(arr))


def compute_observable_occ_roi_cells(plant):
    print("start to compute observation cells")
    start_time = time.time()
    plant = plant.copy() + 1
    shape = np.shape(plant)
    [x1, y1, z1], [x2, y2, z2] = [0, 0, 0], [shape[0], shape[1], shape[2]]
    vertexes = np.array([[x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],
                         [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]])
    left_surface = vertexes[left_surface_index]
    right_surface = vertexes[right_surface_index]
    front_surface = vertexes[front_surface_index]
    back_surface = vertexes[back_surface_index]
    upper_surface = vertexes[upper_surface_index]
    surfaces = [left_surface, right_surface, front_surface, back_surface, upper_surface]
    observable_map = np.zeros_like(plant, np.int)
    xx, yy, zz = np.where(plant >= 2)
    for x, y, z in zip(xx, yy, zz):
        occ_point = np.array([x, y, z])
        for surface in surfaces:
            if not observable_map[x, y, z]:
                observable_map = field_env_3d_helper.check_observable_cell_from_surface(observable_map, plant,
                                                                                        generate_vec3d_from_arr(
                                                                                            occ_point),
                                                                                        generate_vec3d_from_arr(
                                                                                            surface[0]),
                                                                                        generate_vec3d_from_arr(
                                                                                            surface[1]),
                                                                                        generate_vec3d_from_arr(
                                                                                            surface[2]),
                                                                                        generate_vec3d_from_arr(
                                                                                            surface[3]))

    print("# observable cells: ", np.sum(observable_map))
    print("# occupied and roi cells: ", np.sum(plant >= 2))
    print("# roi cells: ", np.sum(plant == 3))
    observable_rois = observable_map[plant == 3]
    print("#observable roi cells:{}; # observable roi rate:{}".format(np.sum(observable_rois),
                                                                      np.sum(observable_rois) / np.sum(plant == 3)))
    observable_occ = observable_map[plant == 2]
    print("#observable occ cells:{}; # observable occ rate:{}".format(np.sum(observable_occ),
                                                                      np.sum(observable_occ) / np.sum(plant == 2)))

    print("compute_observable_occ_roi_cells takes time :{} s".format(time.time() - start_time))

    return observable_map
