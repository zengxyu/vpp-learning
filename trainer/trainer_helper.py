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


def save_episodes_info(phase, episode_info_collector, i_episode, out_result, save_n):
    save_path = os.path.join(out_result, phase + "_log.pkl")
    if i_episode % save_n == 0:
        file = open(save_path, 'wb')
        pickle.dump(episode_info_collector.episode_infos, file)
