#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/11/22 5:51 PM 
    @Description    :
        
===========================================
"""
import os
import pickle
from typing import List, Dict
import numpy as np

from utilities.info import EpisodeInfo


def add_scalar(writer, phase, episode_info, i_episode):
    for key, item in episode_info.items():
        writer.add_scalar(str(phase) + "/" + str(key), item, i_episode)


def add_statistics_to_collector(infos: List[Dict], agent_statistics, episode_info_collector: EpisodeInfo, env):
    # calculate the statistic info for each episode, then added to episode_info_collector
    new_free_cells_sum = 0
    new_occ_cells_sum = 0
    new_roi_cells_sum = 0
    rewards_sum = 0
    visit_gain_sum = 0

    for info in infos:
        visit_gain_sum += info["visit_gain"]
        new_free_cells_sum += info["new_free_cells"]
        new_occ_cells_sum += info["new_occupied_cells"]
        new_roi_cells_sum += info["new_found_rois"]
        rewards_sum += info["reward"]

    print("rewards_sum : ", rewards_sum)
    print("new_free_cells_sum : ", new_free_cells_sum)
    print("new_occ_cells_sum : ", new_occ_cells_sum)
    print("new_roi_cells_sum : ", new_roi_cells_sum)
    print("visit_gain_sum : ", visit_gain_sum)

    print("new_free_cells_rate : ", new_free_cells_sum / env.free_total)
    print("new_occ_cells_rate : ", new_occ_cells_sum / env.occ_total)
    print("new_roi_cells_rate : ", new_roi_cells_sum / env.roi_total)
    print("coverage rate : ", infos[-1]["coverage_rate"])

    episode_info_collector.add({"rewards_sum": rewards_sum})
    episode_info_collector.add({"new_free_cells_sum": new_free_cells_sum})
    episode_info_collector.add({"new_occ_cells_sum": new_occ_cells_sum})
    episode_info_collector.add({"new_roi_cells_sum": new_roi_cells_sum})
    episode_info_collector.add({"visit_gain_sum": visit_gain_sum})

    episode_info_collector.add({"new_free_cells_rate": new_free_cells_sum / env.free_total})
    episode_info_collector.add({"new_occ_cells_rate": new_occ_cells_sum / env.occ_total})
    episode_info_collector.add({"new_roi_cells_rate": new_roi_cells_sum / env.roi_total})
    episode_info_collector.add({"coverage_rate": infos[-1]["coverage_rate"]})

    if not np.isnan(agent_statistics[0][1]):
        episode_info_collector.add({"average_q": agent_statistics[0][1]})
        episode_info_collector.add({"loss": agent_statistics[1][1]})


def save_episodes_info(phase, episode_info_collector, i_episode, parser_args):
    save_path = os.path.join(parser_args.out_folder, phase + "_log.pkl")
    save_n = parser_args.training_config["save_result_n"]
    if i_episode % save_n == 0:
        file = open(save_path, 'wb')
        pickle.dump(episode_info_collector.episode_infos, file)
