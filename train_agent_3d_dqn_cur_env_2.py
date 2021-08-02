import sys

from environment.field_ros import ActionMoRo12
from network.network_dqn import DQN_Network11
from train.RosRandomeTrainer import RosRandomTrainer
from utilities.util import *

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

"""
random starting point, gradually increase the range, fix starting robot rotation direction,
"""
params = {
    'name': 'dqn',

    # model params
    'update_every': 10,
    'eps_start': 0.35,  # Default/starting value of eps
    'eps_decay': 0.99999,  # Epsilon decay rate
    'eps_min': 0.15,  # Minimum epsilon
    'gamma': 0.9,
    'buffer_size': 20000,
    'batch_size': 128,
    'action_size': len(ActionMoRo12),

    'is_double': False,
    'is_priority_buffer': True,

    # grid params
    'max_step': 200,

    # train params
    'is_train': True,
    'visualise': False,
    'is_normalize': False,
    'num_episodes': 5000000,
    'scale': 15,
    'use_gpu': False,
    'network': DQN_Network11,

    # folder params

}

# create folder and return absolute path to log_save_folder, model_save_folder, experience_save_folder, lr_save_folder
params = create_save_folder(params)

# input path
model_in_pth = os.path.join(params['in_folder'], params['model_in'], "Agent_dqn_state_dict_204.mdl")
exp_in_pth = os.path.join(params['in_folder'], params['exp_in'], "buffer.obj")
lr_in_dir = os.path.join(params['in_folder'], params['lr_in'])

randomize_every_episode = 5
trainer = RosRandomTrainer()