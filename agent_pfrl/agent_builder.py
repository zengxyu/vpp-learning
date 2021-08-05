import pfrl
from pfrl import explorers, replay_buffers, agents

import torch
import numpy as np

from agent_pfrl.network import MyCategoryDQN, MyDQN
from network.network_ac_continuous import SAC_PolicyNet3_PFRL, SAC_QNetwork3_PFRL
from network.network_dqn import DQN_Network11_PFRL_Rainbow, DQN_Network11_PFRL


def phi(x):
    frame, robot_pose = x
    frame = frame.astype(np.float32, copy=False)
    robot_pose = robot_pose.astype(np.float32, copy=False)
    return (frame, robot_pose)


def build_dqn_per_agent(action_space, config):
    n_actions = action_space.n
    learning_rate = config.get_learning_rate()
    discount_rate = config.get_discount_rate()
    q_func = DQN_Network11_PFRL(0, n_actions)

    # Use Adam to optimize q_func. eps=1e-2 is for stability.
    optimizer = torch.optim.Adam(q_func.parameters(), lr=learning_rate, eps=1e-2)
    # Set the discount factor that discounts future rewards.

    # Use epsilon-greedy for exploration
    explorer = pfrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.3, random_action_func=action_space.sample)

    # DQN uses Experience Replay.
    # Specify a replay buffer and its capacity.
    replay_buffer = pfrl.replay_buffers.PrioritizedReplayBuffer(capacity=10 ** 6)

    # Set the device id to use GPU. To use CPU only, set it to -1.
    gpu = -1

    # Now create an agent that will interact with the environment.
    agent = pfrl.agents.DoubleDQN(
        q_func,
        optimizer,
        replay_buffer,
        discount_rate,
        explorer,
        replay_start_size=500,
        update_interval=1,
        target_update_interval=100,
        phi=phi,
        gpu=gpu,
    )
    return agent


def build_rainbow_agent(action_space, config):

    n_actions = action_space.n
    learning_rate = config.get_learning_rate()
    discount_rate = config.get_discount_rate()
    decay = config.get_decay()
    n_atoms = config.get_n_atoms()
    q_func = DQN_Network11_PFRL_Rainbow(0, n_actions, n_atoms)
    print("model:", q_func)
    # Noisy nets
    # pfrl.nn.to_factorized_noisy(q_func, sigma_scale=0.1)
    # Turn off explorer decay = 0.99975
    explorer = explorers.LinearDecayEpsilonGreedy(start_epsilon=0.5, end_epsilon=0.1, decay_steps=300 * 400,
                                                  random_action_func=action_space.sample)
    # explorer = explorers.Greedy()
    # Use the same eps as https://arxiv.org/abs/1710.02298
    opt = torch.optim.Adam(q_func.parameters(), lr=learning_rate, eps=1.5e-4)

    # Prioritized Replay
    # Anneal beta from beta0 to 1 throughout training
    update_interval = 1
    n_step_return = 1
    gpu = -1
    replay_start_size = 500
    betasteps = 2 * 10 ** 6 / update_interval
    rbuf = replay_buffers.PrioritizedReplayBuffer(
        10 ** 6,
        alpha=0.5,
        beta0=0.4,
        betasteps=betasteps,
        num_steps=n_step_return,
        normalize_by_max=False,
    )

    agent = MyCategoryDQN(
        q_func,
        opt,
        rbuf,
        gpu=gpu,
        gamma=discount_rate,
        explorer=explorer,
        minibatch_size=32,
        replay_start_size=replay_start_size,
        target_update_interval=2000,
        update_interval=update_interval,
        batch_accumulator="mean",
        phi=phi,
        max_grad_norm=10,
    )
    return agent


def build_multi_ddqn_per(action_space, config):
    # n_actions = action_space.n
    learning_rate = config.get_learning_rate()
    discount_rate = config.get_discount_rate()
    q_func = DQN_Network11_PFRL(0, 15)

    # Use Adam to optimize q_func. eps=1e-2 is for stability.
    optimizer = torch.optim.Adam(q_func.parameters(), lr=learning_rate, eps=1e-2)
    # Set the discount factor that discounts future rewards.

    # Use epsilon-greedy for exploration
    explorer = pfrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.3, random_action_func=action_space.sample)

    # DQN uses Experience Replay.
    # Specify a replay buffer and its capacity.
    replay_buffer = pfrl.replay_buffers.PrioritizedReplayBuffer(capacity=10 ** 6)

    # Set the device id to use GPU. To use CPU only, set it to -1.
    gpu = -1

    # Now create an agent that will interact with the environment.
    agent = pfrl.agents.DoubleDQN(
        q_func,
        optimizer,
        replay_buffer,
        discount_rate,
        explorer,
        replay_start_size=500,
        update_interval=1,
        target_update_interval=100,
        phi=phi,
        gpu=gpu,
    )
    return agent


def build_sac_agent(action_space, config):
    learning_rate_actor = config.get_learning_rate_actor()
    learning_rate_critic = config.get_learning_rate_critic()
    discount_rate = config.get_discount_rate()
    batch_size = config.get_batch_size()
    adam_eps = 1e-1
    action_dim = action_space.low.size
    policy_net = SAC_PolicyNet3_PFRL(state_dim=0, action_dim=action_dim)
    policy_optimizer = torch.optim.Adam(
        policy_net.parameters(), lr=learning_rate_actor, eps=adam_eps
    )
    q_func1 = SAC_QNetwork3_PFRL(state_dim=0, action_dim=action_dim)
    q_func2 = SAC_QNetwork3_PFRL(state_dim=0, action_dim=action_dim)

    q_func1_optimizer = torch.optim.Adam(q_func1.parameters(), lr=learning_rate_critic, eps=adam_eps)
    q_func2_optimizer = torch.optim.Adam(q_func2.parameters(), lr=learning_rate_critic, eps=adam_eps)

    # Prioritized Replay
    # Anneal beta from beta0 to 1 throughout training
    update_interval = 1
    n_step_return = 1
    gpu = -1
    replay_start_size = 500
    rbuf = replay_buffers.ReplayBuffer(10 ** 6, num_steps=n_step_return)

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    agent = pfrl.agents.SoftActorCritic(
        policy_net,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=discount_rate,
        update_interval=update_interval,
        replay_start_size=replay_start_size,
        gpu=gpu,
        minibatch_size=batch_size,
        burnin_action_func=burnin_action_func,
        temperature_optimizer_lr=learning_rate_critic,
    )
    return agent
