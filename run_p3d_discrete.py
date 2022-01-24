import logging

from torch.utils.tensorboard import SummaryWriter
from action_space import ActionMoRo12
from environment.field_p3d_discrete import Field
from load_args import load_dqn_args
from rl_agents.network.network_dqn_11 import DQN_Network11
from rl_agents.pfrl_agents.agent_builder import build_ddqn_agent
from trainer_p3d.P3DTrainer import P3DTrainer
from utilities.basic_logger import setup_logger

setup_logger()
parser_config, config = load_dqn_args()

action_space = ActionMoRo12(n=12)
network = DQN_Network11(action_space.n)

agent = build_ddqn_agent(config, network, action_space)

writer = SummaryWriter()

# load yaml config
env = Field(config=config, action_space=action_space)

learner = P3DTrainer(env=env, agent=agent, action_space=action_space, writer=writer, parser_config=parser_config,
                     training_config=config)
learner.run()
