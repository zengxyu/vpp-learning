import sys
import gym
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from network.network_ac_discrete import SAC_PolicyNet2_Discrete, SAC_QNetwork2_Discrete
from util.summary_writer import SummaryWriterLogger
from utilities.data_structures.Config import Config
from util.util import *

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

env = gym.make('CartPole-v0')

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
config.actor_network = SAC_PolicyNet2_Discrete
config.critic_network = SAC_QNetwork2_Discrete
config.agent = SAC_Discrete
config.is_train = True

config.output_folder = "output_dqn_cart_pole"
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
        "min_steps_before_learning": 50,
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
    "state_size": get_state_size(env),
    "action_size": get_action_size(env, "DISCRETE"),
    "action_shape": get_action_shape(env),
    "action_space": env.action_space
}

config.folder = {
    'out_folder': "output_cartpole_sac",
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

params = create_save_folder(config.folder)

summary_writer = SummaryWriterLogger(config.folder['log_sv'], config.folder['lr_sv'])

player = config.agent(config)


def main_loop():
    time_step = 0

    for i_episode in range(config.num_episodes_to_run):
        print("\nepisode {} start!".format(i_episode))
        done = False
        losses = []
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
            losses.append(loss)
            actions.append(action)
            rewards.append(reward)
            if done:
                # print("mean rewards1:{}".format(np.sum(rewards)))
                # print("actions:{}".format(np.array(actions)))
                # print("rewards:{}".format(np.array(rewards)))
                # print("episode {} over".format(i_episode))
                summary_writer.update(np.mean(losses), np.sum(rewards), i_episode)
                print("episode {} over".format(i_episode))

    print('Complete')


main_loop()
