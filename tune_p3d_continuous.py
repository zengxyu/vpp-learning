import sys
import os
import logging

import action_space
import agents
import network
import ray
from ray import tune

from train.P3DTrainer import P3DTrainer

from config.config_ac import ConfigAC
from environment import field_p3d_continuous
from utilities.util import get_project_path

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


def build_ddpg():
    # network
    Actor_network = network.network_ac_continuous.DDPG_PolicyNet3
    Critic_network = network.network_ac_continuous.DDPG_QNetwork3
    Agent = agents.actor_critic_agents.DDPG.DDPG
    Field = field_p3d_continuous.Field
    Action = action_space.ActionMoRoContinuous

    out_folder = "output_p3d_ddpg"
    in_folder = ""
    return Actor_network, Critic_network, Agent, Field, Action, out_folder, in_folder


def build_sac():
    # network
    Actor_network = network.network_ac_continuous.SAC_PolicyNet3
    Critic_network = network.network_ac_continuous.SAC_QNetwork3
    Agent = agents.actor_critic_agents.SAC.SAC
    Field = field_p3d_continuous.Field
    Action = action_space.ActionMoRoContinuous

    out_folder = "output_p3d_sac"
    in_folder = ""
    return Actor_network, Critic_network, Agent, Field, Action, out_folder, in_folder


def build_sac_per():
    # network
    Actor_network = network.network_ac_continuous.SAC_PER_PolicyNet3
    Critic_network = network.network_ac_continuous.SAC_PER_QNetwork3
    Agent = agents.actor_critic_agents.SAC_Prioritised_Experience_Replay.SAC_Prioritised_Experience_Replay
    Field = field_p3d_continuous.Field
    Action = action_space.ActionMoRoContinuous
    out_folder = "output_p3d_sac_per"
    in_folder = ""
    return Actor_network, Critic_network, Agent, Field, Action, out_folder, in_folder


def build_td3():
    # network
    Actor_network = network.network_ac_continuous.TD3_PolicyNet3
    Critic_network = network.network_ac_continuous.TD3_QNetwork3
    Agent = agents.actor_critic_agents.TD3.TD3
    Field = field_p3d_continuous.Field
    Action = action_space.ActionMoRoContinuous

    out_folder = "output_p3d_td3"
    in_folder = ""
    return Actor_network, Critic_network, Agent, Field, Action, out_folder, in_folder


class ALG:
    DDPG = "DDPG"
    SAC = "SAC"
    SAC_PER = "SAC_PER"
    TD3 = "TD3"


def train_func(tuning_param):
    print("tuning_param:{}".format(tuning_param))

    if tuning_param['alg'] == ALG.SAC:
        Actor_network, Critic_network, Agent, Field, Action, out_folder, in_folder = build_sac()
    elif tuning_param['alg'] == ALG.SAC_PER:
        Actor_network, Critic_network, Agent, Field, Action, out_folder, in_folder = build_sac_per()
    else:
        Actor_network, Critic_network, Agent, Field, Action, out_folder, in_folder = None, None, None, None, None, None, None
        raise NotImplementedError
    # network
    config = ConfigAC(actor_network=Actor_network,
                      critic_network=Critic_network,
                      out_folder=out_folder,
                      in_folder=in_folder,
                      learn_every=1,
                      console_logging_level=logging.DEBUG,
                      file_logging_level=logging.WARNING,
                      )
    config.set_parameters(tuning_param['learning_rate'], tuning_param['discount_rate'])
    trainer = P3DTrainer(config=config, Agent=Agent, Field=Field, Action=Action,
                         project_path=tuning_param["project_path"])

    trainer.train()


def train():
    project_path = get_project_path()

    train_func({"project_path": project_path, "alg": "sac_per", "learning_rate": 1e-2, "discount_rate": 0.9})


if __name__ == '__main__':
    ray.init(local_mode=False)
    project_path = get_project_path()

    print("project path:{}".format(project_path))
    analysis = tune.run(
        train_func,
        config={
            "learning_rate": tune.grid_search([1e-3, 1e-4]),
            "discount_rate": tune.grid_search([0.9, 0.95, 0.98]),
            "project_path": project_path,
            "alg": tune.grid_search([ALG.SAC, ALG.SAC_PER])
        },
        log_to_file=True,
        resources_per_trial={'cpu': 1, 'gpu': 0}
    )
    # train()
