import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import argparse
import time
import numpy as np

from scipy.spatial.transform.rotation import Rotation
from agents.DQN_agents.DDQN_PER import DDQN_PER
from environment.field_ros import Field, ActionMoRo12
from network.network_dqn import DQN_Network11_Dueling
from utilities.summary_writer import MySummaryWriter
from config.config import *

parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=True, action="store_true", help="Run in headless mode")
args = parser.parse_args()
if not args.headless:
    from direct.stdpy import threading

"""
random starting point, gradually increase the range, fix starting robot rotation direction,
"""

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
config.model = DQN_Network11_Dueling
config.agent = DDQN_PER
config.is_train = True

config.output_folder = "output_ros_ddqn_prb_dueling222"
config.log_folder = "log"
config.model_folder = "model"

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 1e-4,
        "batch_size": 128,
        "buffer_size": 40000,
        'exploration_strategy': EpsExplorationStrategy.EXPONENT_STRATEGY,

        "epsilon_decay_rate_denominator": 1,

        "discount_rate": 0.90,
        "tau": 0.01,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 10,
        "linear_hidden_units": [30, 15],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 0.7,
        "learning_iterations": 1,
        "clip_rewards": False,
        'action_size': len(ActionMoRo12)

    },
}

exploration_strategy_config = {EpsExplorationStrategy.INVERSE_STRATEGY: {"epsilon": 1.0,
                                                                      'epsilon_decay_denominator': 1.0},
                               EpsExplorationStrategy.EXPONENT_STRATEGY: {"epsilon": 0.5,
                                                                       "epsilon_decay_rate": 0.997,
                                                                       "epsilon_min": 0.10},
                               EpsExplorationStrategy.CYCLICAL_STRATEGY: {"exploration_cycle_episodes_length": 100}
                               }

config.hyperparameters['DQN_Agents'].update(
    exploration_strategy_config[config.hyperparameters['DQN_Agents']['exploration_strategy']])

config.log_folder = os.path.join(config.output_folder, config.log_folder)
config.model_folder = os.path.join(config.output_folder, config.model_folder)
if not os.path.exists(config.log_folder):
    os.makedirs(config.log_folder)
if not os.path.exists(config.model_folder):
    os.makedirs(config.model_folder)

model_path = ""

summary_writer = MySummaryWriter(config.log_folder)

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=300,
              init_file='VG07_6.binvox', headless=args.headless)

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
            observed_map_next, robot_pose_next, reward, done = field.step(action)

            found_target = reward
            # if robot_pose is the same with the robot_pose_next, then reward--
            if reward == 0:
                zero_reward_consistant_count += 1
            else:
                zero_reward_consistant_count = 0
            # # reward redefine
            if zero_reward_consistant_count >= 10:
                done = True
            robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ initial_direction

            # diff direction
            robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)

            player.step(state=[observed_map, robot_pose_input], action=action, reward=reward,
                        next_state=[observed_map_next, robot_pose_input_next], done=done)

            # to the next state
            observed_map = observed_map_next.copy()
            robot_pose = robot_pose_next.copy()
            # train
            loss = player.learn()

            time_step += 1

            # print(
            #     "{}-th episode : {}-th step takes {} secs; action:{}; found target:{}; sum found targets:{}; reward:{}; sum reward:{}".format(
            #         i_episode,
            #         step_count,
            #         time.time() - time3,
            #         action, found_target, np.sum(found_targets) + found_target, reward,
            #         np.sum(rewards) + reward))
            # record
            summary_writer.add_loss(loss)
            summary_writer.add_reward(found_target, i_episode)

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

                if (i_episode + 1) % 3 == 0:
                    model_save_path = os.path.join(config.model_folder,
                                                   "Agent_dqn_state_dict_%d.mdl" % (i_episode + 1))
                    player.store_model(model_save_path)

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
