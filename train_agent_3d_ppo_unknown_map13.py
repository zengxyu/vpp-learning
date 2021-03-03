import os

import numpy as np
import argparse

from agent.agent_ppo_3d_unknown_map import Agent
from field_env_3d_unknown_map import Field, Action
from memory.GridCellAccessRecord import GridCellAccessRecord
from memory.robot_pose_cluster import RobotPoseCluster
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
    'output': "output_ppo_unknown_map13",
    'config_dir': "config_dir2"
}

if not os.path.exists(params['output']):
    os.mkdir(params['output'])

log_dir = os.path.join(params['output'], 'log')
summary_writer = MySummaryWriter(log_dir)

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=1000,
              init_file='VG07_6.binvox', headless=args.headless)
model_path = os.path.join(params['output'], "Agent_ppo_state_dict_99.mdl")

player = Agent(params, field, summary_writer, train_agent=True, normalize=False, model_path="")

# grid_cell_access_record = GridCellAccessRecord(shape=(256, 256, 256))
robot_pose_cluster = RobotPoseCluster(k=5, field_size=(256, 256, 256), verbose=True)


def main_loop():
    global field, args
    episodes = 200000
    # 可以尝试生成确定的位姿

    for i in range(0, episodes):
        print("\n --- \n \n Episode:{}".format(i))
        done = False
        ts = 0
        rewards = []
        rewards1 = []
        rewards2 = []
        distances_travelled = []
        observed_map, robot_pose = field.reset()
        init_observed_map, init_robot_pose = observed_map, robot_pose

        robot_pose_cluster.add_robot_pose(init_robot_pose[:3])
        destination = robot_pose_cluster.get_destination(init_robot_pose[:3])
        previous_distance_to_destination = np.sqrt(np.sum(np.square(init_robot_pose[:3] - destination)))
        while not done:

            action = player.get_action(observed_map, robot_pose)
            observed_map_prime, robot_pose_prime, reward1, reward3, done = field.step(action)
            if reward1 > 20:
                robot_pose_cluster.add_robot_pose(robot_pose_prime[:3])
            distance_to_destination = np.sqrt(np.sum(np.square(robot_pose_prime[:3] - destination)))
            reward2 = distance_to_destination - previous_distance_to_destination
            print(previous_distance_to_destination, " ", distance_to_destination, " ", reward2)
            # if np.sum(robot_pose[:3] - robot_pose_prime[:3]) != 0:
            #     print("robot pose prime:{}".format(robot_pose_prime))
            # r_ratio = 5
            # reward = reward1 + reward3
            # reward2 = np.log((np.sqrt(np.sum(np.square(robot_pose_prime[:3] - init_robot_pose[:3]))) + 1))
            reward = reward1 + reward2
            # reward = reward1

            rewards1.append(reward1)
            rewards2.append(reward2)
            rewards.append(reward)

            player.store_reward(reward, done)
            summary_writer.add_reward(reward1, i)

            observed_map = observed_map_prime.copy()
            robot_pose = robot_pose_prime.copy()

            if not args.headless:
                threading.Thread.considerYield()

            ts += 1

            if done:
                end_observed_map, end_robot_pose = observed_map, robot_pose
                distance_travelled = np.sqrt(np.sum(np.square(end_robot_pose[:3] - init_robot_pose[:3])))
                distances_travelled.append(distance_travelled)
                # grid_cell_access_record.clear()
                summary_writer.add_episode_len(ts, i)

                robot_pose_cluster.update_cluster()
                print("\nepisode {} over".format(i))
                print("mean rewards1:{}".format(np.sum(rewards1)))
                print("mean rewards2:{}".format(np.sum(rewards2)))
                print("mean rewards:{}".format(np.sum(rewards)))

                print("time steps:{}".format(ts))
                print("learning rate:{}".format(player.optimizer.param_groups[0]['lr']))
                print("distance travelled:{}".format(distance_travelled))
                print("max distance travelled:{}".format(np.max(distances_travelled)))
                print("in this episode, robot travels from {} to {}".format(init_robot_pose[:3], end_robot_pose[:3]))
                # print("mean rewards2:{}; new visit cell num: {}".format(np.sum(rewards2), np.sum(rewards2) / r_ratio))
                player.reset()
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
