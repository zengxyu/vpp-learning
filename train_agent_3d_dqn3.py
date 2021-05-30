import random
import sys
import os
import argparse
import time

from scipy.spatial.transform.rotation import Rotation

from agent.agent_dqn import Agent
from field_ros import Field, Action
from network.network_dqn import DQN_Network6, DQN_Network8, DQN_Network9, DQN_Network11
from util.summary_writer import MySummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=True, action="store_true", help="Run in headless mode")
args = parser.parse_args()
if not args.headless:
    from direct.stdpy import threading

"""
random starting point, gradually increase the range, fix starting robot rotation direction,
"""

params = {
    'name': 'dqn',

    # model params
    'update_every': 10,
    'eps_start': 0.5,  # Default/starting value of eps
    'eps_decay': 0.99999,  # Epsilon decay rate
    'eps_min': 0.15,  # Minimum epsilon
    'gamma': 0.9,
    'buffer_size': 200000,
    'batch_size': 128,
    'action_size': len(Action),

    'is_double': False,
    'is_priority_buffer': True,

    # grid params
    'max_step': 200,

    # train params
    'is_train': True,
    'visualise': True,
    'is_normalize': False,
    'num_episodes': 5000000,
    'scale': 15,
    'use_gpu': False,
    'model': DQN_Network11,

    # folder params

    # output
    'output_folder': "output_dqn2",
    'log_folder': 'log',
    'model_folder': 'model',
    'memory_config_dir': "memory_config",
    'print_info': "small_env"
}

params['log_folder'] = os.path.join(params['output_folder'], params['log_folder'])
params['model_folder'] = os.path.join(params['output_folder'], params['model_folder'])
if not os.path.exists(params['log_folder']):
    os.makedirs(params['log_folder'])
if not os.path.exists(params['model_folder']):
    os.makedirs(params['model_folder'])

# model_path = os.path.join(params['output_folder'], "model", "Agent_dqn_state_dict_1600.mdl")
model_path = os.path.join("output_dqn2", "model", "Agent_dqn_state_dict_81.mdl")

log_dir = os.path.join(params['output_folder'], 'log')
summary_writer = MySummaryWriter(log_dir)

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=300,
              init_file='VG07_6.binvox', headless=args.headless)
player = Agent(params, summary_writer, model_path)

all_mean_rewards = []
all_mean_losses = []


def main_loop():
    time_step = 0

    observed_map, robot_pose = field.reset()
    initial_direction = np.array([[1], [0], [0]])
    print("shape:", observed_map.shape)
    for i_episode in range(params['num_episodes']):
        print("\nInfo:{}; episode {}".format(params['print_info'], i_episode))
        done = False
        rewards1 = []
        actions = []
        e_start_time = time.time()
        step_count = 0
        while not done:
            step_count += 1
            # robot direction
            robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ initial_direction
            robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)

            action = player.act(observed_map, robot_pose_input)
            # action = random.randint(0, 12)
            time3 = time.time()
            observed_map_next, robot_pose_next, reward1, done = field.step(action)
            print(
                "{}-th episode : {}-th step takes {} secs; action:{}; reward:{}".format(i_episode, step_count,
                                                                                        time.time() - time3,
                                                                                        action, reward1))
            found_target = reward1
            # if robot_pose is the same with the robot_pose_next, then reward--
            # if robot_pose == robot_pose_next:
            #     reward1 -= 1

            robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ initial_direction

            # diff direction
            robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)

            player.step(state=[observed_map, robot_pose_input], action=action, reward=reward1,
                        next_state=[observed_map_next, robot_pose_input_next], done=done)

            # to the next state
            observed_map = observed_map_next.copy()
            robot_pose = robot_pose_next.copy()
            # train
            loss = player.learn(memory_config_dir=params['memory_config_dir'])

            time_step += 1
            # record
            summary_writer.add_loss(loss)
            summary_writer.add_reward(found_target, i_episode)

            actions.append(action)
            rewards1.append(int(reward1))

            if not args.headless:
                threading.Thread.considerYield()

            # rewards.append(reward)
            if done:

                print("\nepisode {} over".format(i_episode))
                print("mean rewards1:{}".format(np.sum(rewards1)))
                print("robot pose: {}".format(robot_pose[:3]))
                print("actions:{}".format(np.array(actions)))
                print("rewards:{}".format(np.array(rewards1)))
                # print("mean rewards2:{}; new visit cell num: {}".format(np.sum(rewards2), np.sum(rewards2) / r_ratio))
                player.reset()
                observed_map, robot_pose = field.reset()
                rewards1 = []
                rewards2 = []

                if (i_episode + 1) % 3 == 0:
                    # plt.cla()
                    model_save_path = os.path.join(params['model_folder'],
                                                   "Agent_dqn_state_dict_%d.mdl" % (i_episode + 1))
                    player.store_model(model_save_path)

        e_end_time = time.time()
        print("episode {} spent {} secs".format(i_episode, e_end_time - e_start_time))
    print('Complete')


if args.headless:
    main_loop()
else:
    # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
    # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
    main_thread = threading.Thread(target=main_loop)
    main_thread.start()
    field.gui.run()
