import logging

from config.config import Config
from utilities.data_structures.Constant import EpsExplorationStrategy


class ConfigDQN(Config):
    def __init__(self, network, out_folder, in_folder, learn_every,
                 console_logging_level=logging.WARNING, file_logging_level=logging.WARNING,
                 eps_exploration_strategy_config=None):
        super(ConfigDQN, self).__init__(out_folder, in_folder, learn_every,
                                        console_logging_level=console_logging_level,
                                        file_logging_level=file_logging_level)
        # network config
        self.network = network

        # environment config
        self.environment = {}

        # parameter config
        self.hyper_parameters = None
        self.setup_hyper_parameters()

        self.setup_exploration_strategy_config(eps_exploration_strategy_config)

    def setup_hyper_parameters(self):
        self.hyper_parameters = {
            "DQN_Agents": {
                "learning_rate": 1e-4,
                "batch_size": 128,
                "buffer_size": 8000,
                'eps_exploration_strategy': EpsExplorationStrategy.EXPONENT_STRATEGY,
                "epsilon_decay_rate_denominator": 1,
                "discount_rate": 0.98,
                "tau": 0.01,
                "update_every_n_steps": 10,
                "gradient_clipping_norm": 0.7,
                "learning_iterations": 1,
            }
        }

    def setup_exploration_strategy_config(self, eps_exploration_strategy_config):
        if eps_exploration_strategy_config is not None:
            print("eps_exploration_strategy_config is not None")
            self.hyper_parameters['DQN_Agents'].update(
                eps_exploration_strategy_config[self.hyper_parameters['DQN_Agents']['eps_exploration_strategy']])
        else:
            print("eps_exploration_strategy_config is None")
            self.hyper_parameters['DQN_Agents'].update(
                g_eps_exploration_strategy_config[self.hyper_parameters['DQN_Agents']['eps_exploration_strategy']])

    def set_parameters(self, learning_rate, discount_rate):
        self.hyper_parameters['DQN_Agents']['learning_rate'] = learning_rate
        self.hyper_parameters['DQN_Agents']['discount_rate'] = discount_rate


g_eps_exploration_strategy_config = {EpsExplorationStrategy.INVERSE_STRATEGY: {"epsilon": 1.0,
                                                                               'epsilon_decay_denominator': 1.0},
                                     EpsExplorationStrategy.EXPONENT_STRATEGY: {"epsilon": 0.5,
                                                                                "epsilon_decay_rate": 0.997,
                                                                                "epsilon_min": 0.1},
                                     EpsExplorationStrategy.CYCLICAL_STRATEGY: {
                                         "exploration_cycle_episodes_length": 100}
                                     }
