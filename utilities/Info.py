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
    def __init__(self):
        self.episode_infos = {}
        self.smooth_n = 200

    def add(self, info):
        for key in info.keys():
            if key not in self.episode_infos.keys():
                self.episode_infos[key] = [info[key]]
            else:
                self.episode_infos[key].append(info[key])

    def statistic(self):
        """

        :return:
        """

        statistic_info = {}

        for key, item in self.episode_infos.items():
            statistic_info[key + '_smooth_{}'.format(self.smooth_n)] = np.mean(item[max(len(item) - self.smooth_n, 0):])

        return statistic_info
