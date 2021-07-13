import sys
import argparse
import time

from scipy.spatial.transform.rotation import Rotation

from old_agent.agent_dqn import Agent
from field_ros import Field, Action
from network.network_dqn import DQN_Network11
from utilities.summary_writer import SummaryWriterLogger
from utilities.util import *

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
    'eps_start': 0.35,  # Default/starting value of eps
    'eps_decay': 0.99999,  # Epsilon decay rate
    'eps_min': 0.15,  # Minimum epsilon
    'gamma': 0.9,
    'buffer_size': 4500,
    'batch_size': 128,
    'action_size': len(Action),

    'is_double': False,
    'is_priority_buffer': True,

    # grid params
    'max_step': 200,

    # train params
    'is_train': True,
    'visualise': False,
    'is_normalize': False,
    'num_episodes': 5000000,
    'scale': 15,
    'use_gpu': False,
    'network': DQN_Network11,

    # folder params

    # output
    'out_folder': "output_reset_and_random4",
    'in_folder': "output_reset_and_random3",
    'log_sv': 'log',
    'model_sv': 'model',
    'exp_sv': 'experience',
    'lr_sv': "loss_reward",

    # input
    'model_in': 'model',
    'exp_in': 'experience',
    'lr_in': "loss_reward",

    'print_info': "small_env"
}

# create folder and return absolute path to log_save_folder, model_save_folder, experience_save_folder, lr_save_folder
params = create_save_folder(params)

# input path
model_in_pth = os.path.join(params['in_folder'], params['model_in'], "Agent_dqn_state_dict_60.mdl")
exp_in_pth = os.path.join(params['in_folder'], params['exp_in'], "buffer.obj")
lr_in_dir = os.path.join(params['in_folder'], params['lr_in'])

summary_writer = SummaryWriterLogger(params['log_sv'], params['lr_sv'], lr_in_dir=lr_in_dir)

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=20,
              init_file='VG07_6.binvox', headless=args.headless)
player = Agent(params, summary_writer, model_in_pth=model_in_pth if os.path.exists(model_in_pth) else "",
               exp_in_path=exp_in_pth if os.path.exists(exp_in_pth) else "")

randomize_every_episode = 10


def main_loop():
    time_step = 0
    observed_map, robot_pose = field.reset_and_randomize()
    initial_direction = np.array([[1], [0], [0]])
    print("shape:", observed_map.shape)
    for i_episode in range(params['num_episodes']):
        print("\nInfo:{}; episode {}".format(params['print_info'], i_episode))

        done = False
        losses = []
        rewards = []
        found_targets = []
        actions = []
        e_start_time = time.time()
        step_count = 0
        zero_reward_step_count = 0
        player.reset()

        global randomize_every_episode
        if i_episode % randomize_every_episode == 0:
            observed_map, robot_pose = field.reset_and_randomize()
            randomize_every_episode = max(randomize_every_episode - 1, 5)
        else:
            observed_map, robot_pose = field.reset()
        while not done:
            print("=====================================================================================")
            time3 = time.time()

            step_count += 1
            # robot direction
            robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ initial_direction
            robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)

            action = player.act(observed_map, robot_pose_input)
            observed_map_next, robot_pose_next, reward, done = field.step(action)

            found_target = reward
            # if robot_pose is the same with the robot_pose_next, then reward--
            zero_reward_step_count += 1 if reward == 0 else 0
            done = True if zero_reward_step_count >= 10 else False

            robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ initial_direction

            # diff direction
            robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)

            player.step(state=[observed_map, robot_pose_input], action=action, reward=reward,
                        next_state=[observed_map_next, robot_pose_input_next], done=done)

            # to the next state
            observed_map = observed_map_next.copy()
            robot_pose = robot_pose_next.copy()
            # train
            loss = player.learn()

            print(
                "\n{}-th episode : {}-th step takes {} secs; action:{}; found target:{}; sum found targets:{}; reward:{}; sum reward:{}".format(
                    i_episode,
                    step_count,
                    time.time() - time3,
                    action, found_target, np.sum(found_targets) + found_target, reward,
                    np.sum(rewards) + reward))
            # record
            losses.append(loss)
            actions.append(action)
            rewards.append(reward)
            found_targets.append(int(found_target))
            time_step += 1

            if not args.headless:
                threading.Thread.considerYield()

            if done:
                print("\nepisode {} over".format(i_episode))
                print("robot pose: {}".format(robot_pose[:3]))
                print("actions:{}".format(np.array(actions)))
                print("rewards:{}".format(np.array(rewards)))
                summary_writer.update(np.mean(losses), np.sum(rewards), i_episode)
                # print("mean rewards2:{}; new visit cell num: {}".format(np.sum(rewards2), np.sum(rewards2) / r_ratio))
                if np.sum(rewards) == 0:
                    field.shutdown_environment()
                    field.start_environment()
                    observed_map, robot_pose = field.reset_and_randomize()

                if (i_episode + 1) % 3 == 0:
                    # plt.cla()
                    save_model(player, config.foler, i_episode)

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
