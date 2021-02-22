import os

import numpy as np
import argparse

from agent.agent_ppo_3d_unknown_map import Agent
from field_env_3d_unknown_map import Field, Action
from memory.GridCellAccessRecord import GridCellAccessRecord
from network.network_ppo_3d_unknown_map import PPOPolicy3DUnknownMap2
from util.summary_writer import MySummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=True, action="store_true", help="Run in headless mode")
args = parser.parse_args()

if not args.headless:
    from direct.stdpy import threading

params = {
    'action': Action,

    'traj_collection_num': 16,
    'traj_len': 4,
    'gamma': 0.98,
    'lr': 1e-5,

    'model': PPOPolicy3DUnknownMap2,
    'output': "output_ppo_unknown_map10",
    'config_dir': "config_dir2"
}

if not os.path.exists(params['output']):
    os.mkdir(params['output'])

log_dir = os.path.join(params['output'], 'log')
summary_writer = MySummaryWriter(log_dir)

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=1000,
              init_file='VG07_6.binvox', headless=args.headless)
player = Agent(params, field, summary_writer, train_agent=True, normalize=False)


# grid_cell_access_record = GridCellAccessRecord(shape=(256, 256, 256))


def main_loop():
    global field, args
    episodes = 200000

    observed_map, robot_pose = field.reset()

    for i in range(0, episodes):
        done = False
        ts = 0
        rewards = []
        rewards1 = []
        # rewards2 = []

        while not done:

            action = player.get_action(observed_map, robot_pose)
            observed_map_prime, robot_pose_prime, reward1, reward3, done = field.step(action)

            r_ratio = 5
            # reward2 = grid_cell_access_record.get_reward_of_new_visit(robot_pose_prime) * r_ratio
            # reward = reward1 + reward3

            penalty = 0 if reward1 > 0 else -3
            reward = reward1 + penalty

            rewards1.append(reward1)
            # rewards2.append(reward3)
            rewards.append(reward)

            player.store_reward(reward, done)
            summary_writer.add_reward(reward1, i)

            observed_map = observed_map_prime.copy()
            robot_pose = robot_pose_prime.copy()

            if not args.headless:
                threading.Thread.considerYield()

            ts += 1

            if done:
                player.reset()
                observed_map, robot_pose = field.reset()
                # grid_cell_access_record.clear()
                summary_writer.add_episode_len(ts, i)
                print("\nepisode {} over".format(i))
                print("mean rewards1:{}".format(np.sum(rewards1)))
                print("mean rewards:{}".format(np.sum(rewards)))

                print("time steps:{}".format(ts))
                # print("mean rewards2:{}; new visit cell num: {}".format(np.sum(rewards2), np.sum(rewards2) / r_ratio))

                rewards1 = []
                rewards2 = []

        if (i + 1) % 100 == 0:
            player.store_model(os.path.join(params['output'], "Agent_ppo_state_dict_{}.mdl".format(i)))


if args.headless:
    main_loop()
else:
    # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
    # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
    main_thread = threading.Thread(target=main_loop)
    main_thread.start()
    field.gui.run()
