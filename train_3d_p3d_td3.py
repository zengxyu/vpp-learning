import sys
import os
import time
from scipy.spatial.transform.rotation import Rotation

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import agents
import network
import field_env_3d_unknown_map2_continuous
from utilities.summary_writer import SummaryWriterLogger
from utilities.util import *
from config.config_ac import config

# network
config.actor_network = network.network_ac_continuous.TD3_PolicyNet3
config.critic_network = network.network_ac_continuous.TD3_QNetwork3
config.agent = agents.actor_critic_agents.TD3.TD3
config.field = field_env_3d_unknown_map2_continuous.Field

# output
config.folder['out_folder'] = "output_p3d_td3"
config.folder['in_folder'] = ""
config.folder = create_save_folder(config.folder)
summary_writer = SummaryWriterLogger(config, config.folder['log_sv'], config.folder['lr_sv'])

# Environment : Run in headless mode
headless = True
if not headless:
    from direct.stdpy import threading

field = config.field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=300,
                     init_file='VG07_6.binvox', headless=headless)

config.environment = {
    "reward_threshold": 70000,
    "state_size": None,
    "action_size": 6,
    "action_shape": 6,
}

# Agent
player = config.agent(config)

config.learn_every = 1
config.save_model_every = 50


def main_loop():
    time_step = 0
    initial_direction = np.array([[1], [0], [0]])
    mean_loss_last_n_ep, mean_reward_last_n_ep = 0, 0
    for i_episode in range(config.num_episodes_to_run):
        print("\nepisode {}".format(i_episode))
        e_start_time = time.time()
        done = False
        losses = []
        rewards = []
        actions = []
        player.reset(mean_reward_last_n_ep)
        observed_map, robot_pose = field.reset()
        while not done:
            loss = 0

            # robot direction
            robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ initial_direction
            robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)

            action = player.pick_action([observed_map, robot_pose_input])
            observed_map_next, robot_pose_next, reward, done = field.step(action)

            # if robot_pose is the same with the robot_pose_next, then reward--
            robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ initial_direction

            # diff direction
            robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)

            player.step(state=[observed_map, robot_pose_input], action=action, reward=reward,
                        next_state=[observed_map_next, robot_pose_input_next], done=done)

            # to the next state
            observed_map = observed_map_next.copy()
            robot_pose = robot_pose_next.copy()
            # train
            if time_step % config.learn_every == 0:
                loss = player.learn()

            actions.append(action)
            rewards.append(reward)
            losses.append(loss)
            time_step += 1
            print(
                "{}-th episode :action:{}; reward:{}; sum reward:{}".format(i_episode, action, reward,
                                                                            np.sum(rewards)))
            # record
            if not headless:
                threading.Thread.considerYield()

            if done:
                print("\nepisode {} over".format(i_episode))
                print("robot pose: {}".format(robot_pose[:3]))
                print("actions:{}".format(np.array(actions)))
                print("rewards:{}".format(np.array(rewards)))
                mean_loss_last_n_ep, mean_reward_last_n_ep = summary_writer.update(np.mean(losses), np.sum(rewards),
                                                                                   i_episode)

                # if (i_episode + 1) % config.save_model_every == 0:
                #     save_model(player, config.folder, i_episode)
                e_end_time = time.time()
                print("episode {} spent {} secs".format(i_episode, e_end_time - e_start_time))
    print('Complete')


if headless:
    main_loop()
else:
    # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
    # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
    main_thread = threading.Thread(target=main_loop)
    main_thread.start()
    field.gui.run()
