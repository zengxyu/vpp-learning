import logging
import sys
import os

import ray
from ray import tune

import action_space
import agents
import network
from environment import field_p3d_discrete

from train.P3DTrainer import P3DTrainer

from config.config_dqn import ConfigDQN
from utilities.util import get_project_path

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


class ALG:
    DDQN_PER = "ddqn_per"
    DDQN_DUELING_PER = "ddqn_dueling_per"


def build_ddqn_per():
    Network = network.network_dqn.DQN_Network11
    Agent = agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay.DDQN_With_Prioritised_Experience_Replay
    Field = field_p3d_discrete.Field
    Action = action_space.ActionMoRo12

    out_folder = "output_p3d_ddqn_per"
    in_folder = ""

    return Network, Agent, Field, Action, out_folder, in_folder


def build_ddqn_dueling_per():
    Network = network.network_dqn.DQN_Network11_Dueling
    Agent = agents.DQN_agents.Dueling_DDQN_With_Prioritised_Experience_Replay.Dueling_DDQN_With_Prioritised_Experience_Replay
    Field = field_p3d_discrete.Field
    Action = action_space.ActionMoRo12

    out_folder = "output_p3d_ddqn_dueling"
    in_folder = ""
    return Network, Agent, Field, Action, out_folder, in_folder


def train_fun(tuning_param):
    if tuning_param['alg'] == ALG.DDQN_PER:
        Network, Agent, Field, Action, out_folder, in_folder = build_ddqn_per()
    elif tuning_param['alg'] == ALG.DDQN_DUELING_PER:
        Network, Agent, Field, Action, out_folder, in_folder = build_ddqn_dueling_per()
    else:
        Network, Agent, Field, Action, out_folder, in_folder = None, None, None, None, None, None
        raise NotImplementedError
    # network
    config = ConfigDQN(network=Network,
                       out_folder=out_folder,
                       in_folder=in_folder,
                       learn_every=1,
                       console_logging_level=logging.DEBUG,
                       file_logging_level=logging.WARNING,
                       )
    config.set_parameters(tuning_param['learning_rate'], tuning_param['discount_rate'])

    config.hyper_parameters['DQN_Agents']['learning_rate'] = tuning_param['learning_rate']
    config.hyper_parameters['DQN_Agents']['discount_rate'] = tuning_param['discount_rate']
    trainer = P3DTrainer(config=config, Agent=Agent, Field=Field, Action=Action,
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
            "project_path": project_path,
            "alg": tune.grid_search([ALG.DDQN_DUELING_PER])
        },
        log_to_file=True,
        resources_per_trial={'cpu': 1, 'gpu': 0}
    )
    print("project path:{}".format(project_path))
