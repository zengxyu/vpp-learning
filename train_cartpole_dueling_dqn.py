import sys
import gym
import network
import agents

from config.config_dqn import config
from utilities.summary_writer import SummaryWriterLogger
from utilities.util import *

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

# network
config.network = network.network_dqn.DQN_Network_Dueling_CartPole
config.agent = agents.DQN_agents.DQN.DQN
config.field = 'CartPole-v0'

# output
config.folder['out_folder'] = "output_catpole_dueling_dqn"
config.folder['in_folder'] = ""
config.folder = create_save_folder(config.folder)

# Environment : Run in headless mode
env = gym.make(config.field)
config.environment = {
    "state_size": get_state_size(env),
    "action_size": get_action_size(env, "DISCRETE"),
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


def main_loop():
    time_step = 0

    for i_episode in range(config.num_episodes_to_run):
        print("\nepisode {} start!".format(i_episode))
        done = False
        rewards = []
        actions = []
        losses = []
        state = env.reset()
        player.reset()
        loss = 0
        while not done:
            action = player.pick_action(state)
            state_next, reward, done, _ = env.step(action)
            if done:
                reward = -1
            player.step(state=state, action=action, reward=reward,
                        next_state=state_next, done=done)
            state = state_next
            # train
            if time_step % config.learn_every == 0:
                loss = player.learn()

            time_step += 1
            # record
            losses.append(loss)
            actions.append(action)
            rewards.append(reward)
            if done:
                summary_writer.update(np.mean(losses), np.sum(rewards), i_episode)
                print("episode {} over".format(i_episode))

                if (i_episode + 1) % config.save_model_every == 0:
                    save_model(player, config.folder, i_episode)

    print('Complete')


main_loop()
