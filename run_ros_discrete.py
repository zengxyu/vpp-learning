import logging

from config import process_args
from environment.field_ros import FieldRos
from rl_agents.pfrl_agents.action_builder import build_action_space
from rl_agents.pfrl_agents.agent_builder import build_ddqn_agent
from rl_agents.pfrl_agents.network_builder import build_network
from trainer.RosTrainer import RosTrainer
from utilities.set_random_seed import set_random_seeds

logging.basicConfig(level=logging.ERROR)
# setup_logger()
set_random_seeds(100)

parser_args = process_args("ros")

action_space = build_action_space(parser_args)

network = build_network(parser_args, action_space.n)

agent, _ = build_ddqn_agent(parser_args, network, action_space)

# load yaml config
env = FieldRos(parser_args=parser_args, action_space=action_space)

RosTrainer(env=env, agent=agent, action_space=action_space, parser_args=parser_args).run()
