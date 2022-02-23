import os
import pickle
from typing import List, Dict

import numpy as np


class StepInfo:
    def __init__(self):
        self.step_infos = {}

    def add(self, info):
        """
        info = {"reward":reward, "loss":loss}
        :param info:
        :return:
        """
        for key in info.keys():
            if key not in self.step_infos.keys():
                self.step_infos[key] = [info[key]]
            else:
                self.step_infos[key].append(info[key])

    def statistics(self, key, method="sum"):
        """calculate some statistic information, usually only statistic on sum, for step info"""
        statistic_info = {}
        if key in self.step_infos.keys():
            if method == "sum":
                statistic_info[key + "_sum"] = np.sum(self.step_infos[key])
            else:
                statistic_info[key + "_mean"] = np.mean(self.step_infos[key])
        return statistic_info


class EpisodeInfo:
    def __init__(self, smooth_n):
        self.episode_infos = {}
        self.smooth_n = smooth_n

    def add(self, info):
        for key in info.keys():
            if key not in self.episode_infos.keys():
                self.episode_infos[key] = [info[key]]
            else:
                self.episode_infos[key].append(info[key])

    def get_smooth_statistics(self):
        """

        :return:
        """

        statistic_info = {}

        for key, item in self.episode_infos.items():
            statistic_info[key + '_smooth_{}'.format(self.smooth_n)] = np.mean(item[max(len(item) - self.smooth_n, 0):])

        return statistic_info


class InfoCollector:

    def __init__(self, smooth_n):
        self.episode_infos: Dict[List[List]] = {}
        self.smooth_n = smooth_n
        self.length = 0

    def add(self, episode_info: List[Dict]):
        """
        infos 是什么形式

        """
        self.length += 1
        for i, step_info in enumerate(episode_info):
            for key in step_info.keys():
                if key not in self.episode_infos.keys():
                    self.episode_infos[key] = []
                if i == 0:
                    self.episode_infos[key].append([step_info[key]])
                else:
                    self.episode_infos[key][-1].append(step_info[key])

    def get_ros_smooth_statistic(self, agent_statistics):
        found_roi_cells_sum_latest = np.sum(np.array(self.episode_infos["new_found_rois"])[-1], axis=1)
        found_occ_cells_sum_latest = np.sum(np.array(self.episode_infos["new_occupied_cells"])[-1], axis=1)
        found_free_cells_sum_latest = np.sum(np.array(self.episode_infos["new_free_cells"])[-1], axis=1)
        rewards_sum_latest = np.sum(np.array(self.episode_infos["reward"])[-1], axis=1)
        visit_gain_sum_latest = np.sum(np.array(self.episode_infos["visit_gain"])[-1], axis=1)
        collision_sum_latest = np.sum(np.array(self.episode_infos["collision"])[-1], axis=1)
        coverage_latest = np.array(self.episode_infos["coverage_rate"])[-1][-1]

        print("found_roi_sum : ", found_roi_cells_sum_latest)
        print("found_occ_sum : ", found_occ_cells_sum_latest)
        print("found_free_sum : ", found_free_cells_sum_latest)
        print("rewards_sum : ", rewards_sum_latest)
        print("collision_sum : ", collision_sum_latest)
        print("visit_gain_sum : ", visit_gain_sum_latest)

        result = {}
        result["found_roi_sum"] = found_roi_cells_sum_latest
        result["found_occ_sum"] = found_occ_cells_sum_latest
        result["found_free_sum"] = found_free_cells_sum_latest
        result["rewards_sum"] = rewards_sum_latest
        result["visit_gain_sum"] = visit_gain_sum_latest
        result["collision_sum"] = collision_sum_latest
        result["coverage_rate"] = coverage_latest

        if not np.isnan(agent_statistics[0][1]):
            result["average_q"] = agent_statistics[0][1]
            result["loss"] = agent_statistics[1][1]
        return result

    def get_p3d_smooth_statistic(self, env, agent_statistics):
        left_index = max(self.length - self.smooth_n, 0)
        # compute the sum of the latest n episodes
        found_roi_cells_sum_latest_n = np.sum(np.array(self.episode_infos["new_found_rois"])[left_index:, :], axis=1)
        found_occ_cells_sum_latest_n = np.sum(np.array(self.episode_infos["new_occupied_cells"])[left_index:, :],
                                              axis=1)
        found_free_cells_sum_latest_n = np.sum(np.array(self.episode_infos["new_free_cells"])[left_index:, :], axis=1)
        rewards_sum_latest_n = np.sum(np.array(self.episode_infos["reward"])[left_index:, :], axis=1)
        visit_gain_sum_latest_n = np.sum(np.array(self.episode_infos["visit_gain"])[left_index:, :], axis=1)
        collision_sum_latest_n = np.sum(np.array(self.episode_infos["collision"])[left_index:, :], axis=1)
        coverage_latest_n = np.array(self.episode_infos["coverage_rate"])[left_index:, -1]

        found_roi_cells_sum_latest = found_roi_cells_sum_latest_n[-1]
        found_occ_cells_sum_latest = found_occ_cells_sum_latest_n[-1]
        found_free_cells_sum_latest = found_free_cells_sum_latest_n[-1]
        rewards_sum_latest = rewards_sum_latest_n[-1]
        visit_gain_sum_latest = visit_gain_sum_latest_n[-1]
        collision_sum_latest = collision_sum_latest_n[-1]

        print("found_roi_sum : ", found_roi_cells_sum_latest)
        print("found_occ_sum : ", found_occ_cells_sum_latest)
        print("found_free_sum : ", found_free_cells_sum_latest)
        print("rewards_sum : ", rewards_sum_latest)
        print("visit_gain_sum : ", visit_gain_sum_latest)
        print("collision_sum : ", collision_sum_latest)

        print("found_roi_rate_to_total : {}; found_roi_rate_to_observable : {}".format(
            found_roi_cells_sum_latest / env.roi_total, found_roi_cells_sum_latest / env.observable_roi_total))

        print("found_occ_rate_to_total : {}; found_occ_rate_to_observable : {}".format(
            found_occ_cells_sum_latest / env.occ_total, found_occ_cells_sum_latest / env.observable_occ_total))

        print("found_free_to_total : {};".format(found_free_cells_sum_latest / env.free_total))
        print("rewards : {};".format(rewards_sum_latest))

        found_roi_cells_sum_smooth_n = np.mean(found_roi_cells_sum_latest_n)
        found_occ_cells_sum_smooth_n = np.mean(found_occ_cells_sum_latest_n)
        found_free_cells_sum_smooth_n = np.mean(found_free_cells_sum_latest_n)
        rewards_sum_smooth_n = np.mean(rewards_sum_latest_n)
        visit_gain_sum_smooth_n = np.mean(visit_gain_sum_latest_n)
        collision_sum_smooth_n = np.mean(collision_sum_latest_n)
        coverage_rate_smooth_n = np.mean(coverage_latest_n)

        result = {}
        result["found_roi_sum"] = found_roi_cells_sum_smooth_n
        result["found_occ_sum"] = found_occ_cells_sum_smooth_n
        result["found_free_sum"] = found_free_cells_sum_smooth_n
        result["rewards_sum"] = rewards_sum_smooth_n
        result["visit_gain_sum"] = visit_gain_sum_smooth_n
        result["collision_sum"] = collision_sum_smooth_n

        result["found_roi_rate_to_total"] = found_roi_cells_sum_smooth_n / env.roi_total
        result["found_occ_rate_to_total"] = found_occ_cells_sum_smooth_n / env.occ_total
        result["found_free_rate_to_total"] = found_free_cells_sum_smooth_n / env.free_total
        result["found_roi_rate_to_observable"] = found_roi_cells_sum_smooth_n / env.observable_roi_total
        result["found_occ_rate_to_observable"] = found_occ_cells_sum_smooth_n / env.observable_occ_total
        result["coverage_rate"] = coverage_rate_smooth_n

        if not np.isnan(agent_statistics[0][1]):
            result["average_q"] = agent_statistics[0][1]
            result["loss"] = agent_statistics[1][1]

        return result

    def save_infos(self, phase, i_episode, out_result, save_n):
        save_path = os.path.join(out_result, phase + "_log.pkl")
        if i_episode % save_n == 0:
            file = open(save_path, 'wb')
            pickle.dump(self.episode_infos, file)
