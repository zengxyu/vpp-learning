import numpy as np
import os


def get_eu_distance(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))


def create_save_folder(params):
    # output: to save
    params['model_sv'] = os.path.join(params['out_folder'], params['model_sv'])
    params['exp_sv'] = os.path.join(params['out_folder'], params['exp_sv'])
    params['tb_log_sv'] = os.path.join(params['out_folder'], params['tb_log_sv'])
    params['tb_l_r_sv'] = os.path.join(params['out_folder'], params['tb_l_r_sv'])
    params['bl_log_sv'] = os.path.join(params['out_folder'], params['bl_log_sv'])

    if not os.path.exists(params['model_sv']):
        print("Create folder:{}", params['model_sv'])
        os.makedirs(params['model_sv'])

    if not os.path.exists(params['exp_sv']):
        print("Create folder:{}", params['exp_sv'])
        os.makedirs(params['exp_sv'])

    if not os.path.exists(params['tb_log_sv']):
        print("Create folder:{}", params['tb_log_sv'])
        os.makedirs(params['tb_log_sv'])

    if not os.path.exists(params['tb_l_r_sv']):
        print("Create folder:{}", params['tb_l_r_sv'])
        os.makedirs(params['tb_l_r_sv'])

    if not os.path.exists(params['bl_log_sv']):
        print("Create folder:{}", params['bl_log_sv'])
        os.makedirs(params['bl_log_sv'])

    return params


def save_model(player, params, i_episode):
    player.store_model(model_sv_folder=params['model_sv'], i_episode=i_episode)


def get_state_size(environment):
    """Gets the state_size for the gym env into the correct shape for a neural network"""
    random_state = environment.reset()
    if isinstance(random_state, dict):
        state_size = random_state["observation"].shape[0] + random_state["desired_goal"].shape[0]
        return state_size
    else:
        return random_state.size


def get_action_size(environment, action_types):
    """Gets the action_size for the gym env into the correct shape for a neural network"""
    if "action_size" in environment.__dict__: return environment.action_size
    if action_types == "DISCRETE":
        return environment.action_space.n
    else:
        return environment.action_space.shape[0]


def get_action_shape(environment):
    return environment.action_space.shape
