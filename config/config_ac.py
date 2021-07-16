import logging

from config.Config import Config
from utilities.util import create_save_folder


class ConfigAC(Config):
    def __init__(self, actor_network, critic_network, out_folder, in_folder, learn_every,
                 console_logging_level=logging.WARNING, file_logging_level=logging.WARNING):
        super(ConfigAC, self).__init__(out_folder, in_folder, learn_every,
                                       console_logging_level=console_logging_level,
                                       file_logging_level=file_logging_level)
        # network config
        self.actor_network = actor_network
        self.critic_network = critic_network

        # environment config
        self.environment = None

        # parameter config
        self.hyper_parameters = None
        self.setup_hyper_parameters()

    def setup_output_folder(self, out_folder, in_folder):
        self.folder = {
            'out_folder': out_folder,
            'in_folder': in_folder,

            # output: to save
            'model_sv': 'model',
            'exp_sv': 'experience',
            'tb_l_r_sv': "loss_reward",
            'tb_log_sv': 'tb_log',
            'bl_log_sv': 'bl_log',

            # input
            'model_in': 'model',
            'exp_in': 'experience',
            'tb_l_r_in': "loss_reward",
        }
        self.folder = create_save_folder(self.folder)

    def setup_hyper_parameters(self):
        self.hyper_parameters = {
            "Actor_Critic_Agents": {
                "learning_rate": 0.005,
                "linear_hidden_units": [20, 10],
                "final_layer_activation": ["SOFTMAX", None],
                "gradient_clipping_norm": 5.0,
                "epsilon_decay_rate_denominator": 1.0,
                "normalise_rewards": True,
                "exploration_worker_difference": 2.0,
                "clip_rewards": False,

                "Actor": {
                    "learning_rate": 0.0003,
                    "tau": 0.005,
                    "update_every_n_steps": 10,
                    "gradient_clipping_norm": 5,
                    "initialiser": "Xavier"
                },

                "Critic": {
                    "learning_rate": 0.0003,
                    "tau": 0.005,
                    "update_every_n_steps": 10,
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

# exploration_strategy_config = {ExplorationStrategy.INVERSE_STRATEGY: {"epsilon": 1.0,
#                                                                       'epsilon_decay_denominator': 1.0},
#                                ExplorationStrategy.EXPONENT_STRATEGY: {"epsilon": 0.5,
#                                                                        "epsilon_decay_rate": 0.99999,
#                                                                        "epsilon_min": 0.15},
#                                ExplorationStrategy.CYCLICAL_STRATEGY: {"exploration_cycle_episodes_length": 100}
#                                }

# config.hyperparameters['DQN_Agents'].update(
#     exploration_strategy_config[config.hyperparameters['DQN_Agents']['exploration_strategy']])
