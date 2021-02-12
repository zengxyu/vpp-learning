import numpy as np
from agent.agent_ppo_3d_known_map import Agent
import argparse
from field_env_3d import Field

parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=True, action="store_true", help="Run in headless mode")
args = parser.parse_args()

if not args.headless:
    from direct.stdpy import threading

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=200,
              init_file='VG07_6.binvox', headless=args.headless)
player = Agent(field, train_agent=True)


def main_loop():
    global field, args
    episodes = 200000

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

            print("Timesteps: ", ts)
            print("Reward: ", rew_sum)

            if not args.headless:
                threading.Thread.considerYield()

            ts += 1

            if done:
                total_rewards.append(rew_sum)
                smoothed_rewards.append(np.mean(total_rewards[max(0, i - 200):]))
                print("Timesteps: ", ts)
                print("Reward: ", rew_sum)
                player.reset()
                observed_map, robot_pose = field.reset()
                print("episode {} over".format(i))


if args.headless:
    main_loop()
else:
    main_thread = threading.Thread(target=main_loop)
    main_thread.start()
    field.gui.run()
