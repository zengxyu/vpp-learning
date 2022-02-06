from config import process_args
from environment.field_p3d_discrete import Field
from rl_agents.pfrl_agents.action_builder import build_action_space
from rl_agents.pfrl_agents.agent_builder import build_ddqn_agent
from rl_agents.pfrl_agents.network_builder import build_network
from trainer_p3d.P3DTrainer import P3DTrainer
from utilities.basic_logger import setup_logger

setup_logger()

parser_args = process_args()

action_space = build_action_space(parser_args)

network = build_network(parser_args, action_space.n)

agent = build_ddqn_agent(parser_args, network, action_space)

# load yaml config
env = Field(parser_args=parser_args, action_space=action_space)

P3DTrainer(env=env, agent=agent, action_space=action_space, parser_args=parser_args).run()
