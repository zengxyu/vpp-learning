# import matplotlib.pyplot as plt
# import pickle
import numpy as np
# from agent_ppo import Agent
from random_agent_3d import RandomAgent
import argparse
from direct.stdpy import threading

from field_env_3d import Field
import time

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=200, init_file='VG07_6.binvox', headless=args.headless)


def main_loop():
    global field
    episodes = 200000

    # player = Agent(field, train_agent = True)
    player = RandomAgent(field)

    print('resetting field')
    observed_map, robot_pose = field.reset()
    print('field resetted')

    total_rewards, smoothed_rewards = [], []

    for i in range(0, episodes):
        done = False
        ts = 0
        rew_sum = 0
        while not done:
            action = player.get_action(observed_map, robot_pose)
            observed_map, robot_pose, reward, done = field.step(action)
            player.store_reward(reward, done)

            rew_sum += reward

            ts += 1

            if done:
                total_rewards.append(rew_sum)
                smoothed_rewards.append(np.mean(total_rewards[max(0, i-200):]))
                print("Timesteps: ", ts)
                print("Reward: ", rew_sum)
                player.reset()
                observed_map, robot_pose = field.reset()
                print("episode {} over".format(i))

                # if (i+1) % 1000 == 0:
                #     player.store_model("Agent_ppo_state_dict_%d.mdl" % (i+1))
                #     plt.plot(total_rewards)
                #     plt.plot(smoothed_rewards)
                #     plt.title("Total reward per episode")
                #     plt.savefig("rewards_ppo_%d_episodes.png" % (i+1))
                #     plt.clf()

    # save dict
    # player.store_model("Agent_ppo_state_dict.mdl")
    # plt.plot(total_rewards)
    # plt.plot(smoothed_rewards)
    # plt.title("Total reward per episode")
    # plt.savefig("rewards.png")


if args.headless:
    main_loop()
else:
    # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
    # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
    main_thread = threading.Thread(target=main_loop)
    main_thread.start()
    field.gui.run()
