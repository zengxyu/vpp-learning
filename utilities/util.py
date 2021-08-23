import numpy as np
import os


def compute_conv_out_width(i, k, s, p):
    """
    计算卷积输出层的宽度
    :param i: 输入尺寸
    :param k: 卷积核大小
    :param s: 步幅
    :param p: 边界扩充
    :return: 输出的feature map的宽
    """
    o = (i - k + 2 * p) / s + 1
    return int(o)


def compute_de_conv_out_width(i, k, s, p):
    """
    计算反卷积输出层的宽度
    :param i: 输入尺寸
    :param k: 卷积核大小
    :param s: 步幅
    :param p: 边界扩充
    :return: 输出的feature map的宽
    """
    out = (i - 1) * s + k - 2 * p
    return int(out)


def compute_conv_out_node_num(d, w, h):
    """
    计算卷积后输出层的节点数量
    :param d: depth channel number
    :param w: width
    :param h: height
    :return:
    """
    return int(d * w * h)


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


def build_in_folder(params):
    params['model_in'] = os.path.join(params['in_folder'], params['model_in'])
    params['exp_in'] = os.path.join(params['in_folder'], params['exp_in'])
    params['tb_l_r_in'] = os.path.join(params['in_folder'], params['tb_l_r_in'])
    return params


def save_model(player, params, i_episode):
    player.store_model(model_sv_folder=params['model_sv'], i_episode=i_episode)


def get_state_size(environment):
    """Gets the state_size for the gym environment into the correct shape for a neural network"""
    random_state = environment.reset()
    if isinstance(random_state, dict):
        state_size = random_state["observation"].shape[0] + random_state["desired_goal"].shape[0]
        return state_size
    else:
        return random_state.size


def get_action_size(environment, action_types):
    """Gets the action_size for the gym environment into the correct shape for a neural network"""
    if "action_size" in environment.__dict__: return environment.action_size
    if action_types == "DISCRETE":
        return environment.action_space.n
    else:
        return environment.action_space.shape[0]


def get_action_shape(environment):
    return environment.action_space.shape


def get_project_path():
    cur_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return cur_dir


if __name__ == '__main__':
    # print(get_project_path())
    a = compute_conv_out_width(i=9, k=4, s=2, p=1)
    b = compute_conv_out_width(i=18, k=4, s=2, p=1)

    print(a, b)
    c = compute_conv_out_node_num(9, 18, 50)
    print(c)
