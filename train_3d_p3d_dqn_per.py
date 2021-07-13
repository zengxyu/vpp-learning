import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import agents
import network
import field_env_3d_unknown_map2
from utilities.summary_writer import SummaryWriterLogger
from utilities.util import *
from config.config_dqn import config
import numpy as np
from scipy.spatial.transform.rotation import Rotation

config.network = network.network_dqn.DQN_Network11
config.agent = agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay.DDQN_With_Prioritised_Experience_Replay
config.field = field_env_3d_unknown_map2.Field

# output
config.folder['out_folder'] = "output_p3d_ddqn_per"
config.folder['in_folder'] = ""
config.folder = create_save_folder(config.folder)
summary_writer = SummaryWriterLogger(config, config.folder['log_sv'], config.folder['lr_sv'])

# Environment : Run in headless mode
headless = True
if not headless:
    from direct.stdpy import threading
field = config.field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=300,
                     init_file='VG07_6.binvox', headless=headless, is_augment_env=False)

config.environment = {
    "is_vpp": True,
    "action_size": len(field.actions),
    "action_space": field.actions,
}

# Agent
player = config.agent(config)

config.learn_every = 1
config.save_model_every = 50


def main_loop():
    time_step = 0
    initial_direction = np.array([[1], [0], [0]])

    for i_episode in range(config.num_episodes_to_run):
        print("episode {} start!".format(i_episode))
        done = False
        losses = []
        rewards = []
        actions = []
        player.reset()
        observed_map, robot_pose = field.reset()

        while not done:
            # robot direction
            loss = 0

            robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ initial_direction
            robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)

            action = player.pick_action([observed_map, robot_pose_input])
            observed_map_next, robot_pose_next, reward, _, done = field.step(action)

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

            # record
            losses.append(loss)
            rewards.append(reward)
            actions.append(action)
            time_step += 1

            if not headless:
                threading.Thread.considerYield()

            if done:
                summary_writer.update(np.mean(losses), np.sum(rewards), i_episode)

                print("\nepisode {} over".format(i_episode))
                print("mean rewards1:{}".format(np.sum(rewards)))
                print("robot pose ends in: {}".format(robot_pose[:3]))
                print("actions:{}".format(np.array(actions)))
                print("rewards:{}".format(np.array(rewards)))

                if (i_episode + 1) % config.save_model_every == 0:
                    # plt.cla()
                    save_model(player, config.folder, i_episode)

    print('Complete')


if headless:
    main_loop()
else:
    # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
    # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
    main_thread = threading.Thread(target=main_loop)
    main_thread.start()
    field.gui.run()
