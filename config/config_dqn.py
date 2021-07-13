from utilities.data_structures.Config import Config
from utilities.data_structures.Constant import EpsExplorationStrategy

config = Config()
config.seed = 1
config.num_episodes_to_run = 10000
config.use_GPU = False
config.is_train = True
config.debug_mode = False

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 1e-4,
        "batch_size": 128,
        "buffer_size": 40000,
        'eps_exploration_strategy': EpsExplorationStrategy.EXPONENT_STRATEGY,
        "epsilon_decay_rate_denominator": 1,
        "discount_rate": 0.90,
        "tau": 0.01,
        "update_every_n_steps": 10,
        "gradient_clipping_norm": 0.7,
        "learning_iterations": 1,
    },
}
config.folder = {
    'out_folder': None,
    'in_folder': None,

    # output: to save
    'log_sv': 'log',
    'model_sv': 'model',
    'exp_sv': 'experience',
    'lr_sv': "loss_reward",

    # input
    'model_in': 'model',
    'exp_in': 'experience',
    'lr_in': "loss_reward",
}

eps_exploration_strategy_config = {EpsExplorationStrategy.INVERSE_STRATEGY: {"epsilon": 1.0,
                                                                             'epsilon_decay_denominator': 1.0},
                                   EpsExplorationStrategy.EXPONENT_STRATEGY: {"epsilon": 0.5,
                                                                              "epsilon_decay_rate": 0.99999,
                                                                              "epsilon_min": 0.15},
                                   EpsExplorationStrategy.CYCLICAL_STRATEGY: {"exploration_cycle_episodes_length": 100}
                                   }

config.hyperparameters['DQN_Agents'].update(
    eps_exploration_strategy_config[config.hyperparameters['DQN_Agents']['exploration_strategy']])
