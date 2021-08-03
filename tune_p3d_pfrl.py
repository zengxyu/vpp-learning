import logging
import sys
import os

import ray
from ray import tune

import action_space
from agent_pfrl.agent_type import AgentType
from environment import field_p3d_discrete, field_p3d_continuous

from config.config_dqn import ConfigDQN
from train_pfrl.P3DTrainer_PFRL import P3DTrainer_PFRL
from utilities.util import get_project_path

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


def build_rainbow():
    Field = field_p3d_discrete.Field
    Action = action_space.ActionMoRo12
    out_folder = "output_p3d_rainbow"
    in_folder = ""

    return Field, Action, out_folder, in_folder


def build_ddqn_per():
    Field = field_p3d_discrete.Field
    Action = action_space.ActionMoRo12
    out_folder = "output_p3d_ddqn_per"
    in_folder = ""

    return Field, Action, out_folder, in_folder


def build_sac():
    Field = field_p3d_continuous.Field
    Action = action_space.ActionMoRoContinuous
    out_folder = "output_p3d_sac"
    in_folder = ""

    return Field, Action, out_folder, in_folder


def train_fun(tuning_param):
    if tuning_param['alg'] == AgentType.Agent_Rainbow:
        Field, Action, out_folder, in_folder = build_rainbow()
    elif tuning_param['alg'] == AgentType.Agent_DDQN_PER:
        Field, Action, out_folder, in_folder = build_ddqn_per()
    elif tuning_param['alg'] == AgentType.Agent_SAC:
        Field, Action, out_folder, in_folder = build_sac()
    else:
        Field, Action, out_folder, in_folder = None, None, None, None
    # network
    config = ConfigDQN(network=None,
                       out_folder=out_folder,
                       in_folder=in_folder,
                       learn_every=1,
                       console_logging_level=logging.DEBUG,
                       file_logging_level=logging.WARNING,
                       )
    config.set_parameters(tuning_param)
    trainer = P3DTrainer_PFRL(config=config, agent_type=tuning_param['alg'], Field=Field, Action=Action,
                              project_path=tuning_param["project_path"])

    trainer.train()


if __name__ == '__main__':
    ray.init(local_mode=False)
    project_path = get_project_path()
    analysis = tune.run(
        train_fun,
        config={
            "learning_rate": tune.grid_search([1e-3, 1e-4]),
            "discount_rate": tune.grid_search([0.9, 0.95, 0.98]),
            "epsilon_decay_rate": tune.grid_search([0.99, 0.99975, 0.9999999]),
            # "learning_rate": tune.grid_search([1e-4]),
            # "discount_rate": tune.grid_search([0.98]),
            "project_path": project_path,
            "alg": tune.grid_search([AgentType.Agent_Rainbow])
        },
        log_to_file=True,
        resources_per_trial={'cpu': 1, 'gpu': 0}
    )
    print("project path:{}".format(project_path))
