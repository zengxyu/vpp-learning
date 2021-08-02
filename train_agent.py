import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
from old_agent.agent_ppo import Agent
import argparse

from environment.field_2d import Field

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

if not args.headless:
    pg.init()

field = Field(shape=(64, 64), target_count=100, sensor_range=5, scale=8, max_steps=200, headless=args.headless)
episodes = 200000

player = Agent(field, train_agent = True)

observed_map, robot_pose = field.reset()

total_rewards, smoothed_rewards = [], []
for i in range(0,episodes):
    done = False
    ts = 0
    rew_sum = 0
    while not done:
        action = player.get_action(observed_map, robot_pose)
        observed_map, robot_pose, reward, done = field.step(action)
        player.store_reward(reward, done)

        rew_sum += reward

        ts += 1

        if not args.headless:
            pg.event.get()

        if done:
            total_rewards.append(rew_sum)
            smoothed_rewards.append(np.mean(total_rewards[max(0, i-200):]))
            print("Timesteps: ", ts)
            print("Reward: ", rew_sum)
            player.reset()
            observed_map, robot_pose = field.reset()
            print("episode {} over".format(i))

            if (i+1) % 1000 == 0:
                player.store_model("Agent_ppo_state_dict_%d.mdl" % (i+1))
                plt.plot(total_rewards)
                plt.plot(smoothed_rewards)
                plt.title("Total reward per episode")
                plt.savefig("rewards_ppo_%d_episodes.png" % (i+1))
                plt.clf()

            

# save dict
player.store_model("Agent_ppo_state_dict.mdl")
plt.plot(total_rewards)
plt.plot(smoothed_rewards)
plt.title("Total reward per episode")
plt.savefig("rewards.png")

if not args.headless:
    pg.quit()



