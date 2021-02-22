import os

from memory.RewardedPoseRecord import RewardedPoseRecord

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np
import argparse

from agent.agent_ppo_3d_unknown_map import Agent
from field_env_3d_unknown_map import Field, Action
from memory.GridCellAccessRecord import GridCellAccessRecord
from util.summary_writer import MySummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=True, action="store_true", help="Run in headless mode")
args = parser.parse_args()

if not args.headless:
    from direct.stdpy import threading

params = {
    'action': Action,

    'traj_collection_num': 64,
    'traj_len': 4,
    'gamma': 0.98,
    'lr': 1e-5,

    'output': "output_ppo_unknown_map4",
    'config_dir': "config_dir2"
}

if not os.path.exists(params['output']):
    os.mkdir(params['output'])

log_dir = os.path.join(params['output'], 'log')
summary_writer = MySummaryWriter(log_dir)

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=1000,
              init_file='VG07_6.binvox', headless=args.headless)
player = Agent(params, field, summary_writer, train_agent=True, normalize=True)

grid_cell_access_record = GridCellAccessRecord(shape=(256, 256, 256))

pose_record = RewardedPoseRecord()


def main_loop():
    global field, args
    episodes = 200000

    observed_map, robot_pose = field.reset()
    c_ratio = 0.1
    for i in range(0, episodes):
        done = False
        ts = 0
        rewards1 = []
        rewards2 = []
        rewards3 = []

        while not done:

            action = player.get_action(observed_map, robot_pose)
            observed_map_prime, robot_pose_prime, reward1, done = field.step(action)

            if reward1 > 0:
                pose_record.put_pose(robot_pose_prime)

            reward3 = c_ratio * pose_record.get_reward(robot_pose_prime)
            # print("reward3:{}".format(reward3))
            reward = reward1 + reward3

            rewards1.append(reward1)
            # rewards2.append(reward2)
            rewards3.append(reward3)

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
                grid_cell_access_record.clear()

                print("\nepisode {} over".format(i))
                print("mean rewards1:{}".format(np.sum(rewards1)))
                # print("mean rewards2:{}; new visit cell num: {}".format(np.sum(rewards2), np.sum(rewards2) / r_ratio))
                print("mean rewards3:{}".format(np.sum(rewards3)))

                print("mean reward sum :{}".format(np.sum(rewards1) + np.sum(rewards3)))
                rewards1 = []
                rewards2 = []
                rewards3 = []


if args.headless:
    main_loop()
else:
    # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
    # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
    main_thread = threading.Thread(target=main_loop)
    main_thread.start()
    field.gui.run()
