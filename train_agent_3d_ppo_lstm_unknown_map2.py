import os
import argparse

import torch

from agent.agent_ppo_lstm_3d_unknown_map import Agent
from field_env_3d_unknown_map import Field, Action
from network.network_ppo_lstm import PPO_LSTM2
from util.summary_writer import MySummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=True, action="store_true", help="Run in headless mode")
args = parser.parse_args()

if not args.headless:
    from direct.stdpy import threading

params = {
    'traj_collection_num': 16,
    'traj_len': 4,
    'gamma': 0.98,
    'lr': 1e-5,

    'batch_size': 16,
    'seq_len': 4,
    'num_layers': 1,
    'model': PPO_LSTM2,  # PPO_LSTM, PPO_LSTM2, PPO_LSTM3
    'action_size': len(Action),
    'output': "output_ppo_lstm_unknown_map2"

}

if not os.path.join(params['output']):
    os.mkdir(params['output'])

log_dir = os.path.join(params['output'], 'log')
summary_writer = MySummaryWriter(log_dir)

field = Field(shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=1000,
              init_file='VG07_6.binvox', headless=args.headless)
player = Agent(params, summary_writer)


def main_loop():
    global field, args
    episodes = 200000

    for i in range(0, episodes):
        observed_map, robot_pose = field.reset()

        h_out = torch.zeros([params['num_layers'], 1, 32], dtype=torch.float)
        c_out = torch.zeros([params['num_layers'], 1, 32], dtype=torch.float)

        done = False
        while not done:
            h_in, c_in = h_out, c_out
            action, value, probs, h_out, c_out = player.act(observed_map, robot_pose, h_in, c_in)
            observed_map_prime, robot_pose_prime, reward, done = field.step(action.detach().cpu().numpy()[0][0])
            player.store_data(
                [observed_map, robot_pose, action.detach().cpu().numpy().squeeze(), reward, observed_map_prime,
                 robot_pose_prime, value.detach().cpu().numpy().squeeze(), probs.detach().cpu().numpy().squeeze(),
                 done, h_in.detach().cpu().numpy(), c_in.detach().cpu().numpy(), h_out.detach().cpu().numpy(),
                 c_out.detach().cpu().numpy()])

            observed_map = observed_map_prime.copy()
            robot_pose = robot_pose_prime.copy()

            summary_writer.add_reward(reward, i)

            if not args.headless:
                threading.Thread.considerYield()

            if done:
                print("\nepisode {} over".format(i))

            if player.memory.is_full_batch():
                loss = player.train_net()

                player.memory.reset_data()

                summary_writer.add_loss(loss)


if args.headless:
    main_loop()
else:
    main_thread = threading.Thread(target=main_loop)
    main_thread.start()
    field.gui.run()
