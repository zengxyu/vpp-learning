import os

import numpy as np
import argparse

from agent.agent_ppo_3d_unknown_map import Agent
from field_env_3d_unknown_map import Field, Action
from util.summary_writer import MySummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=True, action="store_true", help="Run in headless mode")
args = parser.parse_args()

if not args.headless:
    from direct.stdpy import threading

params = {
    'action': Action,

    'traj_collection_num': 4,
    'traj_len': 4,
    'gamma': 0.98,
    'lr': 1e-5,

    'output': "output_ppo_unknown_map"

}

if not os.path.join(params['output']):
    os.mkdir(params['output'])

log_dir = os.path.join(params['output'], 'log')
summary_writer = MySummaryWriter(log_dir)

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=200,
              init_file='VG07_6.binvox', headless=args.headless)
player = Agent(params, field, summary_writer, train_agent=True)


def main_loop():
    global field, args
    episodes = 200000

    observed_map, robot_pose = field.reset()

    for i in range(0, episodes):
        done = False
        ts = 0
        while not done:
            action = player.get_action(observed_map, robot_pose)
            observed_map, robot_pose, reward, done = field.step(action)
            player.store_reward(reward, done)

            summary_writer.add_reward(reward, i)

            if not args.headless:
                threading.Thread.considerYield()

            ts += 1

            if done:
                player.reset()
                observed_map, robot_pose = field.reset()
                print("\nepisode {} over".format(i))


if args.headless:
    main_loop()
else:
    # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
    # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
    main_thread = threading.Thread(target=main_loop)
    main_thread.start()
    field.gui.run()
