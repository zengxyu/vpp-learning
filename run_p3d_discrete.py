from torch.utils.tensorboard import SummaryWriter
from action_space import ActionMoRo10
from environment.field_p3d_discrete import Field
from load_args import load_dqn_args
from rl_agents.network.network_attention import SpatialAttentionModel, SpatialAttentionModel2
from rl_agents.pfrl_agents.agent_builder import build_ddqn_agent
from trainer_p3d.P3DTrainer import P3DTrainer
from utilities.basic_logger import setup_logger

setup_logger()
parser_config, training_config = load_dqn_args()

action_space = ActionMoRo10(n=10)
# network = DQN_Network11(action_space.n)
network = SpatialAttentionModel2(n_actions=action_space.n)
agent = build_ddqn_agent(training_config, network, action_space)

writer = SummaryWriter()

# load yaml config
env = Field(config=training_config, action_space=action_space)

learner = P3DTrainer(env=env, agent=agent, action_space=action_space, writer=writer, parser_config=parser_config,
                     training_config=training_config)
learner.run()
