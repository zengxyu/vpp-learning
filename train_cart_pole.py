import sys
import os
import gym
import numpy as np

from agents.DQN_agents.DDQN import DDQN
from agents.DQN_agents.DQN import DQN
from agents.DQN_agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets
from agents.DQN_agents.Dueling_DDQN import Dueling_DDQN
from network.network_dqn import DQN_Network6, DQN_Network8, DQN_Network9, DQN_Network11, DQN_Network_CartPole, \
    DQN_Network_Dueling_CartPole
from util.summary_writer import MySummaryWriter
from utilities.data_structures.Config import Config

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

config = Config()
config.seed = 1
config.num_episodes_to_run = 450
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
# Dueling DQN should use DQN_Network_Dueling_CartPole
config.model = DQN_Network_Dueling_CartPole
config.agent = DQN
config.is_train = True

config.output_folder = "output_dqn_cart_pole"
config.log_folder = "log"
config.model_folder = "model"

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "buffer_size": 40000,
        "epsilon": 1.0,
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
        'action_size': 2

    },
    "Stochastic_Policy_Search_Agents": {
        "policy_network_type": "Linear",
        "noise_scale_start": 1e-2,
        "noise_scale_min": 1e-3,
        "noise_scale_max": 2.0,
        "noise_scale_growth_factor": 2.0,
        "stochastic_action_decision": False,
        "num_policies": 10,
        "episodes_per_policy": 1,
        "num_policies_to_keep": 5,
        "clip_rewards": False
    },
    "Policy_Gradient_Agents": {
        "learning_rate": 0.05,
        "linear_hidden_units": [20, 20],
        "final_layer_activation": "SOFTMAX",
        "learning_iterations_per_round": 5,
        "discount_rate": 0.99,
        "batch_norm": False,
        "clip_epsilon": 0.1,
        "episodes_per_learning_round": 4,
        "normalise_rewards": True,
        "gradient_clipping_norm": 7.0,
        "mu": 0.0,  # only required for continuous action games
        "theta": 0.0,  # only required for continuous action games
        "sigma": 0.0,  # only required for continuous action games
        "epsilon_decay_rate_denominator": 1.0,
        "clip_rewards": False
    },

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
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
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

config.log_folder = os.path.join(config.output_folder, config.log_folder)
config.model_folder = os.path.join(config.output_folder, config.model_folder)
if not os.path.exists(config.log_folder):
    os.makedirs(config.log_folder)
if not os.path.exists(config.model_folder):
    os.makedirs(config.model_folder)

summary_writer = MySummaryWriter(config.log_folder)

env = gym.make('CartPole-v0')
player = config.agent(config)


def main_loop():
    time_step = 0

    for i_episode in range(config.num_episodes_to_run):
        print("\nepisode {} start!".format(i_episode))
        done = False
        rewards = []
        actions = []
        state = env.reset()
        player.reset()

        while not done:
            action = player.pick_action(state)
            state_next, reward, done, _ = env.step(action)
            if done:
                reward = -1
            player.step(state=state, action=action, reward=reward,
                        next_state=state_next, done=done)
            state = state_next
            # train
            loss = player.learn()

            time_step += 1
            # record
            summary_writer.add_loss(loss)
            summary_writer.add_reward(reward, i_episode)

            actions.append(action)
            rewards.append(reward)
            if done:
                print("mean rewards1:{}".format(np.sum(rewards)))
                print("actions:{}".format(np.array(actions)))
                print("rewards:{}".format(np.array(rewards)))
                print("episode {} over".format(i_episode))

                if (i_episode + 1) % 50 == 0:
                    model_save_path = os.path.join(config.model_folder,
                                                   "Agent_dqn_state_dict_%d.mdl" % (i_episode + 1))
                    player.store_model(model_save_path)

    print('Complete')


main_loop()
