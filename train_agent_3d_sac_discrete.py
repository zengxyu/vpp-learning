import sys
import os

from field_env_3d_unknown_map2 import Field, Action

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import argparse
import time
import numpy as np

from scipy.spatial.transform.rotation import Rotation
from utilities.summary_writer import MySummaryWriter
from config.config import *
from network.network_ac_discrete import *
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete

parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=True, action="store_true", help="Run in headless mode")
args = parser.parse_args()
if not args.headless:
    from direct.stdpy import threading

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=300,
              init_file='VG07_6.binvox', headless=args.headless)


def get_state_size(field):
    observed_map, robot_pose = field.reset()
    return observed_map.shape


config = Config()
config.seed = 1
config.num_episodes_to_run = 40000
config.file_to_save_data_results = "results/data_and_graphs/Cart_Pole_Results_Data.pkl"
config.file_to_save_results_graph = "results/data_and_graphs/Cart_Pole_Results_Graph.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False

config.actor_network = SAC_PolicyNet3_Discrete
config.critic_network = SAC_QNetwork3_Discrete

config.agent = SAC_Discrete
config.is_train = True

config.output_folder = "output_sac_discrete_001"
config.log_folder = "log"
config.model_folder = "model"

config.hyperparameters = {
    "Actor_Critic_Agents": {
        "learning_rate": 0.005,
        "linear_hidden_units": [20, 10],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
        "buffer_size": 1000000,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}

config.environment = {
    "state_size": get_state_size(field),
    "action_size": len(Action),
    "action_shape": len(Action),
    "action_space": np.random.randint(0, len(Action) - 1)
}

config.folder = {
    'out_folder': "output_sac_discrete_001",
    'in_folder': "",
    'log_sv': 'log',
    'model_sv': 'model',
    'exp_sv': 'experience',
    'lr_sv': "loss_reward",

    # input
    'model_in': 'model',
    'exp_in': 'experience',
    'lr_in': "loss_reward",
}

config.log_folder = os.path.join(config.output_folder, config.log_folder)
config.model_folder = os.path.join(config.output_folder, config.model_folder)
if not os.path.exists(config.log_folder):
    os.makedirs(config.log_folder)
if not os.path.exists(config.model_folder):
    os.makedirs(config.model_folder)

model_path = ""

summary_writer = MySummaryWriter(config.log_folder)

player = config.agent(config)

all_mean_rewards = []
all_mean_losses = []


def main_loop():
    time_step = 0

    observed_map, robot_pose = field.reset()
    initial_direction = np.array([[1], [0], [0]])
    print("shape:", observed_map.shape)
    for i_episode in range(config.num_episodes_to_run):
        print("\nepisode {}".format(i_episode))
        done = False
        rewards = []
        found_targets = []
        actions = []
        e_start_time = time.time()
        step_count = 0
        zero_reward_consistant_count = 0
        player.reset()
        observed_map, robot_pose = field.reset()
        while not done:
            step_count += 1
            # robot direction
            robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ initial_direction
            robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)

            action = player.pick_action([observed_map, robot_pose_input])
            time3 = time.time()
            observed_map_next, robot_pose_next, reward, _, done = field.step(action)

            found_target = reward
            # if robot_pose is the same with the robot_pose_next, then reward--
            robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ initial_direction

            # diff direction
            robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)

            player.step(state=[observed_map, robot_pose_input], action=action, reward=reward,
                        next_state=[observed_map_next, robot_pose_input_next], done=done)

            # to the next state
            observed_map = observed_map_next.copy()
            robot_pose = robot_pose_next.copy()
            if step_count % 3 == 0:
                loss = player.learn()
                summary_writer.add_loss(loss)
            # train
            summary_writer.add_reward(found_target, i_episode)

            time_step += 1

            # print(
            #     "{}-th episode : {}-th step takes {} secs; action:{}; found target:{}; sum found targets:{}; reward:{}; sum reward:{}".format(
            #         i_episode,
            #         step_count,
            #         time.time() - time3,
            #         action, found_target, np.sum(found_targets) + found_target, reward,
            #         np.sum(rewards) + reward))
            # record

            actions.append(action)
            rewards.append(reward)
            found_targets.append(int(found_target))
            if not args.headless:
                threading.Thread.considerYield()

            if done:
                print("\nepisode {} over".format(i_episode))
                print("mean rewards1:{}".format(np.sum(rewards)))
                print("robot pose: {}".format(robot_pose[:3]))
                print("actions:{}".format(np.array(actions)))
                print("rewards:{}".format(np.array(rewards)))

                # if (i_episode + 1) % 3 == 0:
                #     model_save_path = os.path.join(config.model_folder,
                #                                    "Agent_dqn_state_dict_%d.mdl" % (i_episode + 1))
                #     player.store_model(model_save_path)

                e_end_time = time.time()
                print("episode {} spent {} secs".format(i_episode, e_end_time - e_start_time))
    print('Complete')


if args.headless:
    main_loop()
else:
    # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
    # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
    main_thread = threading.Thread(target=main_loop)
    main_thread.start()
    field.gui.run()
