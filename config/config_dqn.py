import logging

from config.Config import Config
from utilities.data_structures.Constant import EpsExplorationStrategy


class ConfigDQN(Config):
    def __init__(self, network, out_folder, in_folder, learn_every,
                 console_logging_level=logging.WARNING, file_logging_level=logging.WARNING):
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
        self.setup_exploration_strategy_config()

    def setup_hyper_parameters(self):
        self.hyper_parameters = {
            "DQN_Agents": {
                "learning_rate": 1e-4,
                "batch_size": 128,
                "buffer_size": 10000,
                'eps_exploration_strategy': EpsExplorationStrategy.EXPONENT_STRATEGY,
                "epsilon_decay_rate_denominator": 1,
                "discount_rate": 0.98,
                "tau": 0.01,
                "update_every_n_steps": 10,
                "gradient_clipping_norm": 0.7,
                "learning_iterations": 1,
            }
        }

    def setup_exploration_strategy_config(self):
        self.hyper_parameters['DQN_Agents'].update(
            eps_exploration_strategy_config[self.hyper_parameters['DQN_Agents']['eps_exploration_strategy']])


eps_exploration_strategy_config = {EpsExplorationStrategy.INVERSE_STRATEGY: {"epsilon": 1.0,
                                                                             'epsilon_decay_denominator': 1.0},
                                   EpsExplorationStrategy.EXPONENT_STRATEGY: {"epsilon": 0.5,
                                                                              "epsilon_decay_rate": 0.9997,
                                                                              "epsilon_min": 0.1},
                                   EpsExplorationStrategy.CYCLICAL_STRATEGY: {"exploration_cycle_episodes_length": 100}
                                   }
