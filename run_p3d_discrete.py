from agents.baseline_agents.baseline_agent_builder import build_baseline_agent
from agents.policies import Policies
from config import process_args
from environment.field_p3d import FieldP3D
from agents.pfrl_agents.action_builder import build_action_space
from agents.pfrl_agents.agent_builder import build_ddqn_agent
from agents.pfrl_agents.network_builder import build_network
from trainer.P3DTrainer import P3DTrainer
from utilities.basic_logger import setup_logger
from utilities.set_random_seed import set_random_seeds

setup_logger()
set_random_seeds(100)

parser_args = process_args("p3d")
action_space = build_action_space(parser_args)

# load yaml config
env = FieldP3D(parser_args=parser_args, action_space=action_space)

network = build_network(parser_args, action_space.n)

agent, scheduler = build_ddqn_agent(parser_args, network, action_space)

P3DTrainer(env=env, agent=agent, scheduler=scheduler, action_space=action_space, parser_args=parser_args).run()
