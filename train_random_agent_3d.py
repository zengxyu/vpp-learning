import os
import argparse
from old_agent.random_agent_3d import RandomAgent
from environment.field_env_3d_unknown_map import Field, Action
from utilities.summary_writer import MySummaryWriter
from memory.NormalizationSaver import NormalizationSaver

parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=False, action="store_true", help="Run in headless mode")
args = parser.parse_args()

params = {
    'action': Action,

    'traj_collection_num': 16,
    'traj_len': 4,
    'gamma': 0.98,
    'lr': 1e-5,

    'output': "output_random_agent_3d",
    'config_dir': "config_dir"
}

if not os.path.exists(params['output']):
    os.mkdir(params['output'])

if not os.path.exists(params['config_dir']):
    print("create config_dir")
    os.mkdir(params['config_dir'])

log_dir = os.path.join(params['output'], 'log')
summary_writer = MySummaryWriter(log_dir)

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=1000,
              init_file='VG07_6.binvox', headless=args.headless)
player = RandomAgent(field)

normalization_saver = NormalizationSaver()

if not args.headless:
    from direct.stdpy import threading

def main_loop():
    global field, args
    episodes = 200000

    for i in range(0, episodes):
        done = False
        ts = 0
        observed_map, robot_pose = field.reset()
        while not done:
            action = player.get_action(observed_map, robot_pose)
            observed_map_prime, robot_pose_prime, reward1, reward3, done = field.step(action)

            normalization_saver.store_state_and_reward(observed_map, robot_pose, reward1, reward3, done)

            observed_map = observed_map_prime.copy()
            robot_pose = robot_pose_prime.copy()

            ts += 1

            if done:
                player.reset()
                observed_map, robot_pose = field.reset()
                print("\nepisode {} over".format(i))

        if (i + 1) % 50 == 0:
            print("save to file")
            normalization_saver.save_2_local(params['config_dir'])


if args.headless:
    main_loop()
else:
    main_thread = threading.Thread(target=main_loop)
    main_thread.start()
    field.gui.run()
