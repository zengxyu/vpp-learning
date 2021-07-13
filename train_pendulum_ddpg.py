import sys
import gym
import agents
import network
from utilities.summary_writer import *
from utilities.util import *
from config.config_ac import config

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

"""收敛：在episode=40的时候开始出现收敛的迹象, reward上升不太稳定"""

# network
config.actor_network = network.network_ac_continuous.DDPG_PolicyNet
config.critic_network = network.network_ac_continuous.DDPG_QNetwork
config.agent = agents.actor_critic_agents.DDPG.DDPG
config.field = 'Pendulum-v0'

# output
config.folder['out_folder'] = "output_pendulum_ddpg"
config.folder['in_folder'] = ""
config.folder = create_save_folder(config.folder)

# Environment : Run in headless mode
env = gym.make(config.field)
config.environment = {
    "reward_threshold": 0,
    "state_size": get_state_size(env),
    "action_size": get_action_size(env, "CONTINUOUS"),
    "action_shape": get_action_shape(env),
    "action_space": env.action_space
}

# Agent
player = config.agent(config)

# frequency
config.learn_every = 1
config.save_model_every = 50

# summary writer
summary_writer = SummaryWriterLogger(config, config.folder['log_sv'], config.folder['lr_sv'])

action_range = [env.action_space.low, env.action_space.high]


def main_loop():
    time_step = 0
    mean_reward_last_n_ep = 0
    for i_episode in range(config.num_episodes_to_run):
        print("\nepisode {} start!".format(i_episode))
        done = False
        losses = []
        rewards = []
        actions = []
        state = env.reset()
        player.reset(mean_reward_last_n_ep)
        loss = 0

        while not done:

            action = player.pick_action(state)
            action_in = (action - (-1)) * (action_range[1] - action_range[0]) / 2.0 + action_range[0]
            state_next, reward, done, _ = env.step(action_in)
            if done:
                reward = -1
            player.step(state=state, action=action, reward=reward,
                        next_state=state_next, done=done)
            state = state_next
            # train
            if time_step % config.learn_every == 0:
                loss = player.learn()

            # record
            losses.append(loss)
            rewards.append(reward)
            actions.append(action)

            time_step += 1

            if done:
                _, mean_reward_last_n_ep = summary_writer.update(np.mean(losses), np.sum(rewards), i_episode)
                print("episode {} over".format(i_episode))

    print('Complete')


main_loop()
