import pickle
import numpy as np

total_num = 75370


def generate_eval_rewards(path, label):
    _, rewards_random_env = pickle.load(open(path, 'rb'))
    rewards_random_env_avg = np.mean(rewards_random_env)
    print(
        "{} average count:{}; average percentage:{}; len:{}".format(label,
                                                                    rewards_random_env_avg,
                                                                    rewards_random_env_avg / total_num,
                                                                    len(rewards_random_env)))


def generate_eval_rewards_seq_len_5():
    # len = 5
    path5 = "/Users/weixianshi/PycharmProjects/vpp-learning/output/out_p3d_random_step_len_5_36_action_predict_model_440/loss_reward/loss_reward.obj"
    # path6 = "/Users/weixianshi/PycharmProjects/vpp-learning/output_evaluation_step_len_5/out_p3d_random_step_len_5_36_action_predict_keep_training_model_440/loss_reward/loss_reward.obj"
    # path7 = "/Users/weixianshi/PycharmProjects/vpp-learning/output/out_p3d_static_step_len_5_36_action_predict_model_275/loss_reward/loss_reward.obj"
    # path8 = "/Users/weixianshi/PycharmProjects/vpp-learning/output/out_p3d_static_step_len_5_36_action_predict_model_275_keep_training/loss_reward/loss_reward.obj"

    # random exploration
    path9 = "/Users/weixianshi/PycharmProjects/vpp-learning/output_evaluation_step_len_5/out_p3d_random_predict/loss_reward/loss_reward.obj"
    # path10 = "/Users/weixianshi/PycharmProjects/vpp-learning/output/out_p3d_static_env_random_select_predict/loss_reward/loss_reward.obj"

    # # diagonal
    # path11 = "/Users/weixianshi/PycharmProjects/vpp-learning/output_evaluation_step_len_5/out_p3d_z_random_env_predict/loss_reward/loss_reward.obj"
    # path12 = "/Users/weixianshi/PycharmProjects/vpp-learning/output/out_p3d_static_env_z_predict/loss_reward/loss_reward.obj"

    # len = 10
    path13 = "/Users/weixianshi/PycharmProjects/vpp-learning/output/out_p3d_random_step_len_10_36_action_predict_model_550/loss_reward/loss_reward.obj"
    # path14 = "/Users/weixianshi/PycharmProjects/vpp-learning/output/out_p3d_random_step_len_10_36_action_predict_model_550_keep_training/loss_reward/loss_reward.obj"
    # path15 = "/Users/weixianshi/PycharmProjects/vpp-learning/output/out_p3d_static_step_len_10_36_action_predict_model_550/loss_reward/loss_reward.obj"
    # path16 = "/Users/weixianshi/PycharmProjects/vpp-learning/output/out_p3d_static_step_len_10_36_action_predict_model_550_keep_training/loss_reward/loss_reward.obj"

    print()
    generate_eval_rewards(path5, "rewards_random_env_without_training_step_len_5")
    # generate_eval_rewards(path6, "rewards_random_env_with_training_step_len_5")
    # generate_eval_rewards(path7, "rewards_static_env_without_training_step_len_5")
    # generate_eval_rewards(path8, "rewards_static_env_with_training_step_len_5")
    print()
    generate_eval_rewards(path9, "rewards_random_env_random_exploration")
    # generate_eval_rewards(path10, "rewards_static_env_random_exploration")
    print()
    # generate_eval_rewards(path11, "rewards_random_env_diagonal_scanning")
    # generate_eval_rewards(path12, "rewards_static_env_diagonal_scanning")
    print()
    generate_eval_rewards(path13, "rewards_random_env_without_training_step_len_10")
    # generate_eval_rewards(path14, "rewards_random_env_with_training_step_len_10")
    # generate_eval_rewards(path15, "rewards_static_env_without_training_step_len_10")
    # generate_eval_rewards(path16, "rewards_static_env_with_training_step_len_10")
    """
    rewards_random_env_without_training average count:38159.35; average percentage:0.5062936181504577
    rewards_random_env_with_training average count:39929.8; average percentage:0.5297837335810004
    rewards_static_env_without_training average count:29390.6; average percentage:0.3899509088496749
    rewards_static_env_with_training average count:34111.55; average percentage:0.4525878996948388
    rewards_random_env average count:12014.75; average percentage:0.15941024280217594
    rewards_static_env average count:28080.45; average percentage:0.37256799787713946
    rewards_random_env_diagonal_scanning average count:10147.0; average percentage:0.1346291627968688
    rewards_static_env_diagonal_scanning average count:26970.0; average percentage:0.35783468223431075
    """


if __name__ == '__main__':
    generate_eval_rewards_seq_len_5()
