import os
import pickle

import numpy as np

from utilities.util import get_project_path

total_num = 75370


def compute_reward(path, label):
    rewards, found_targets = pickle.load(open(path, "rb"))
    print("{}; reward mean of an episode:{}; found target mean of an episode:{}; coverage rate:{}; len:{}".format(label,
                                                                                                                  np.mean(
                                                                                                                      rewards),
                                                                                                                  np.mean(
                                                                                                                      found_targets),
                                                                                                                  np.mean(
                                                                                                                      found_targets) / total_num,
                                                                                                                  len(rewards)))


def compute_time(path, label):
    taken_steps = pickle.load(open(path, "rb"))
    print("{}; taken steps:{}".format(label, np.mean(taken_steps)))


if __name__ == '__main__':
    path_reward_len_5 = os.path.join(get_project_path(),
                                     "output/out_p3d_random_step_len_5_36_action_predict_model_440/reward_found_targets.obj")
    path_reward_len_10 = os.path.join(get_project_path(),
                                      "output/out_p3d_random_step_len_10_36_action_predict_model_550/reward_found_targets.obj")
    path_reward_random_exploration = os.path.join(get_project_path(),
                                                  "output/out_p3d_random_predict/reward_found_targets.obj")
    path_reward_diagonal_scanning = os.path.join(get_project_path(),
                                                 "output/out_p3d_random_env_z_predict/reward_found_targets.obj")

    compute_reward(path_reward_len_5, "seq_len = 5:")
    compute_reward(path_reward_len_10, "seq_len = 10:")
    compute_reward(path_reward_random_exploration, "random_exploration:")
    compute_reward(path_reward_diagonal_scanning, "diagonal_scanning:")

    print()
    path_taken_steps_len_5 = os.path.join(get_project_path(),
                                          "output/out_p3d_random_step_len_5_36_action_predict_model_440_statistic_time/steps.obj")
    path_taken_steps_len_10 = os.path.join(get_project_path(),
                                           "output/out_p3d_random_step_len_10_36_action_predict_model_550_statistic_time/steps.obj")
    path_taken_steps_random_exploration = os.path.join(get_project_path(),
                                                       "output/out_p3d_random_predict_statistic_time/steps.obj")
    compute_time(path_taken_steps_len_5, "seq_len = 5:")
    compute_time(path_taken_steps_len_10, "seq_len = 10:")
    compute_time(path_taken_steps_random_exploration, "random_exploration:")
