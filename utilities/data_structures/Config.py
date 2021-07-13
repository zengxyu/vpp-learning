from utilities.data_structures.Constant import EpsExplorationStrategy


class Config(object):
    """Object to hold the config requirements for an agent/game"""

    def __init__(self):
        self.seed = None
        self.environment = None
        self.agent = None

        self.hyperparameters = None
        self.folder = None

        self.learn_every = None
        self.save_model_every = None

        self.use_GPU = None
        self.debug_mode = None
        self.is_train = None

        self.num_episodes_to_run = None
