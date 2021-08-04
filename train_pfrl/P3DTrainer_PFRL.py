import os
import time

from scipy.spatial.transform.rotation import Rotation

from agent_pfrl.agent_builder import *
from agent_pfrl.agent_type import AgentType
from utilities.basic_logger import BasicLogger
from utilities.summary_writer import SummaryWriterLogger

headless = True
if not headless:
    from direct.stdpy import threading




class P3DTrainer_PFRL(object):
    def __init__(self, config, agent_type, Field, Action, project_path):
        self.config = config
        self.Field = Field
        self.Action = Action

        # field
        init_file_path = os.path.join(project_path, 'VG07_6.binvox')
        self.field = self.Field(Action=Action, shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05,
                                max_steps=300, init_file=init_file_path, headless=headless)
        self.config.environment = {
            "is_vpp": True,
            "reward_threshold": 0,
            "state_size": self.get_state_size(self.field),
            "action_size": self.get_action_size(self.field),
            "action_shape": self.get_action_size(self.field),
        }
        # agent
        if agent_type == AgentType.Agent_Rainbow:
            self.agent = build_rainbow_agent(self.field.action_space, config)
        elif agent_type == AgentType.Agent_DDQN_PER:
            self.agent = build_dqn_per_agent(self.field.action_space, config)
        elif agent_type == AgentType.Agent_SAC:
            self.agent = build_sac_agent(self.field.action_space, config)
        elif agent_type ==AgentType.Agent_Multi_DDQN_PER:
            self.agent = build_multi_ddqn_per(self.field.action_space, config)

        else:
            raise NotImplementedError
        self.summary_writer = SummaryWriterLogger(config)
        self.logger = BasicLogger.setup_console_logging(config)

    def train(self):
        if headless:
            self.main_loop()
        else:
            # field.gui.taskMgr.setupTaskChain('mainTaskChain', numThreads=1)
            # field.gui.taskMgr.add(main_loop, 'mainTask', taskChain='mainTaskChain')
            main_thread = threading.Thread(target=self.main_loop)
            main_thread.start()
            self.field.gui.run()

    def main_loop(self):
        time_step = 0
        initial_direction = np.array([[1], [0], [0]])
        mean_loss_last_n_ep, mean_reward_last_n_ep = 0, 0
        for i_episode in range(self.config.num_episodes_to_run):
            print("\nepisode {}".format(i_episode))
            e_start_time = time.time()
            done = False
            losses = []
            rewards = []
            actions = []
            observed_map, robot_pose = self.field.reset()
            t = 0  # time step

            while not done:
                loss = 0
                t += 1

                # robot direction
                robot_direction = Rotation.from_quat(robot_pose[3:]).as_matrix() @ initial_direction
                robot_pose_input = np.concatenate([robot_pose[:3], robot_direction.squeeze()], axis=0)

                action = self.agent.act((observed_map, robot_pose_input))

                (observed_map_next, robot_pose_next), reward, done, _ = self.field.step(action)

                # if robot_pose is the same with the robot_pose_next, then reward--
                robot_direction_next = Rotation.from_quat(robot_pose_next[3:]).as_matrix() @ initial_direction

                # diff direction
                robot_pose_input_next = np.concatenate([robot_pose_next[:3], robot_direction_next.squeeze()], axis=0)

                self.agent.observe([observed_map, robot_pose_input], reward, done, done)

                # to the next state
                observed_map = observed_map_next.copy()
                robot_pose = robot_pose_next.copy()

                actions.append(action)
                rewards.append(reward)
                losses.append(loss)
                time_step += 1
                # print(
                #     "{}-th episode : {}-th step takes {} secs; action:{}; found target:{}; sum found targets:{}; reward:{}; sum reward:{}".format(
                #         i_episode,
                #         step_count,
                #         time.time() - time3,
                #         action, found_target, np.sum(found_targets) + found_target, reward,
                #         np.sum(rewards) + reward))
                # record
                if not headless:
                    threading.Thread.considerYield()

                if done:
                    print("\nepisode {} over".format(i_episode))
                    print("robot pose: {}".format(robot_pose[:3]))
                    print("actions:{}".format(np.array(actions)))
                    print("rewards:{}".format(np.array(rewards)))
                    mean_loss_last_n_ep, mean_reward_last_n_ep = self.summary_writer.update(np.mean(losses),
                                                                                            np.sum(rewards),
                                                                                            i_episode)
                    print("statistic:{}".format(self.agent.get_statistics()))
                    # if (i_episode + 1) % self.config.save_model_every == 0:
                    #     self.agent.store_model()

                    e_end_time = time.time()
                    print("episode {} spent {} secs".format(i_episode, e_end_time - e_start_time))
        print('Complete')

    def get_state_size(self, field):
        return 0

    def get_action_size(self, field):
        return field.get_action_size()
