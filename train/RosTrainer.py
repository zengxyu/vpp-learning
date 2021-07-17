import numpy as np
import time
from scipy.spatial.transform.rotation import Rotation

from utilities.basic_logger import BasicLogger
from utilities.summary_writer import SummaryWriterLogger

headless = True
if not headless:
    from direct.stdpy import threading


class RosTrainer(object):
    def __init__(self, config, Agent, Field, Action):
        self.config = config
        self.Agent = Agent
        self.Field = Field
        self.summary_writer = SummaryWriterLogger(config)
        self.logger = BasicLogger.setup_console_logging(config)

        self.field = Field(Action=Action, shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, max_steps=300,
                           handle_simulation=False)
        self.config.environment = {
            "is_vpp": True,
            "reward_threshold": 0,
            "state_size": 0,
            "action_size": self.field.get_action_size(),
            "action_shape": self.field.get_action_size()
        }
        self.agent = Agent(config)

    def train(self):
        self.main_loop()

    def main_loop(self):
        time_step = 0
        initial_direction = np.array([[1], [0], [0]])
        for i_episode in range(self.config.num_episodes_to_run):
            print("\nepisode {}".format(i_episode))
            step_count = 0
            e_start_time = time.time()
            done = False
            rewards = []
            actions = []
            losses = []
            zero_reward_consistent_count = 0
            self.agent.reset()
            observed_map, robot_pose = self.field.reset()
            while not done:
                start_step_time = time.time()
                # robot direction
                robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ initial_direction
                robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)

                action = self.agent.pick_action([observed_map, robot_pose_input])
                observed_map_next, robot_pose_next, reward, done = self.field.step(action)

                # if robot_pose is the same with the robot_pose_next, then reward--
                if reward == 0:
                    zero_reward_consistent_count += 1
                else:
                    zero_reward_consistent_count = 0
                # # reward redefine
                if zero_reward_consistent_count >= 10:
                    done = True
                robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ initial_direction

                # diff direction
                robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)

                self.agent.step(state=[observed_map, robot_pose_input], action=action, reward=reward,
                                next_state=[observed_map_next, robot_pose_input_next], done=done)

                # to the next state
                observed_map = observed_map_next.copy()
                robot_pose = robot_pose_next.copy()
                # train
                loss = self.agent.learn()

                time_step += 1
                step_count += 1
                print(
                    "{}-th episode : {}-th step takes {} secs; reward:{}; sum reward:{}".format(
                        i_episode,
                        step_count,
                        time.time() - start_step_time,
                        reward,
                        np.sum(rewards) + reward))
                # record

                actions.append(action)
                rewards.append(reward)
                losses.append(loss)

                if not headless:
                    threading.Thread.considerYield()

                if done:
                    print("\nepisode {} over".format(i_episode))
                    print("mean rewards1:{}".format(np.sum(rewards)))
                    print("robot pose: {}".format(robot_pose[:3]))
                    print("actions:{}".format(np.array(actions)))
                    print("rewards:{}".format(np.array(rewards)))
                    mean_loss_last_n_ep, mean_reward_last_n_ep = self.summary_writer.update(np.mean(losses),
                                                                                            np.sum(rewards),
                                                                                            i_episode)
                    if (i_episode + 1) % 3 == 0:
                        self.agent.store_model()

                    e_end_time = time.time()
                    print("episode {} spent {} secs".format(i_episode, e_end_time - e_start_time))
        print('Complete')
