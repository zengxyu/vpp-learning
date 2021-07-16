from utilities.util import create_save_folder


class Config(object):
    """Object to hold the config requirements for an agent/game"""

    def __init__(self, out_folder, in_folder, learn_every, console_logging_level, file_logging_level):
        self.seed = 1

        # train config
        self.use_GPU = False
        self.is_train = True
        self.debug_mode = False

        # number of frequency config
        self.num_episodes_to_run = 10000
        self.learn_every = learn_every
        self.save_model_every = 50
        self.tb_save_l_r_every_n_episode = 1
        self.tb_smooth_l_r_every_n_episode = 50
        self.bl_console_logging_level = console_logging_level
        self.bl_file_logging_level = file_logging_level

        # folder config
        self.folder = None
        self.setup_output_folder(out_folder, in_folder)

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
