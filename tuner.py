import configparser
import logging
import os

import pfrl
import gym
import numpy
import ray
import torch
from ray import tune

from crowd_sim import register_env, CrowdEnv
from crowd_sim.envs.utils.robot import Robot
import dqn_trainer
from util.agent_helper import build_dqn_agent, build_rainbow_agent
from tensorboardX import SummaryWriter

from util.utils import get_device


def train_fun(config):
    logging.basicConfig(level="INFO")

    out_dir = "out_" + str(config['visible'])
    model_dir = os.path.join(out_dir, "model")

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "log_train"))

    env_config = config['env_config']
    robot = Robot(env_config, 'robot')
    robot.set_visible(config['visible'])

    env: CrowdEnv = config['environment']
    env.set_robot(robot)
    env.set_with_om(config['with_om'])

    # Set a random seed used in PFRL.
    pfrl.utils.set_random_seed(42)
    phi = lambda x: x.astype(numpy.float32, copy=False)

    gpu = config['gpu']
    device = get_device(gpu)

    # agent = build_dqn_agent(environment, phi, gpu, device)
    agent = build_rainbow_agent(env, phi, gpu=gpu, device=device, n_atoms=config['n_atoms'])
    dqn_trainer.train(writer, env, agent, 4500 + config['n_atoms'] * 25, model_dir)

    # test
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "log_eval"))
    model_dir = os.path.join(model_dir, "epi_4000")
    dqn_trainer.eval_agent(writer, env, agent, 5000, False, model_dir)


if __name__ == '__main__':
    # 设为True方便调试，但是只能跑一个
    ray.init(local_mode=False)

    env_config = configparser.RawConfigParser()
    env_config.read("configs/environment.config")
    # GPU的device_id(如：0)，只使用CPU则设为-1
    gpu = 0 if torch.cuda.is_available() else -1

    register_env()
    env = gym.make('CrowdSim-v0', config=env_config)

    analysis = tune.run(
        train_fun,
        config={
            "environment": env,
            "gpu": gpu,
            "env_config": env_config,
            "n_atoms": tune.grid_search([10, 30, 51]),
            "visible": False,
            "with_om": tune.grid_search([True, False])
        },
        log_to_file=True,
        resources_per_trial={'cpu': 2, 'gpu': 0.2}
    )
