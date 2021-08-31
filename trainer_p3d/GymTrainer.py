import gym
import numpy as np

from utilities.action_type import ActionType
from utilities.basic_logger import BasicLogger
from utilities.summary_writer import SummaryWriterLogger


class GymTrainer(object):
    def __init__(self, config, Agent, Field, action_type):
        self.config = config
        self.Agent = Agent
        self.Field = Field
        self.summary_writer = SummaryWriterLogger(config)
        self.logger = BasicLogger.setup_console_logging(config)
        self.field = gym.make(self.Field)

        config.environment = {
            "reward_threshold": 0,
            "state_size": self.get_state_size(self.field),
            "action_size": self.get_action_size(self.field, action_type),
            "action_shape": self.get_action_shape(self.field),
            "action_space": self.field.action_space
        }

        self.agent = self.Agent(self.config)

    def train(self):
        # Agent
        time_step = 0

        for i_episode in range(self.config.num_episodes_to_run):
            print("\nepisode {} start!".format(i_episode))
            done = False
            losses = []
            rewards = []
            actions = []
            state = self.field.reset()
            self.agent.reset()
            loss = 0

            while not done:
                action = self.agent.pick_action(state)
                action_in = self.action_transform(self.field, action)
                state_next, reward, done, _ = self.field.step(action_in)
                if done:
                    reward = -1
                self.agent.step(state=state, action=action, reward=reward,
                                next_state=state_next, done=done)
                state = state_next
                # trainer_p3d
                if time_step % self.config.learn_every == 0:
                    loss = self.agent.learn()

                # record
                losses.append(loss)
                rewards.append(reward)
                actions.append(action)

                time_step += 1

                if done:
                    self.summary_writer.update(np.mean(losses), np.sum(rewards), i_episode)
                    if (i_episode + 1) % self.config.save_model_every == 0:
                        self.agent.store_model()
                    print("episode {} over".format(i_episode))

        print('Complete')

    def action_transform(self, env, action):
        action_in = action
        if self.Field == "Pendulum-v0":
            action_range = [env.action_space.low, env.action_space.high]

            action_in = (action - (-1)) * (action_range[1] - action_range[0]) / 2.0 + action_range[0]
        return action_in

    def get_state_size(self, field):
        """Gets the state_size for the gym environment into the correct shape for a neural network"""
        random_state = field.reset()
        if isinstance(random_state, dict):
            state_size = random_state["observation"].shape[0] + random_state["desired_goal"].shape[0]
            return state_size
        else:
            return random_state.size

    def get_action_size(self, field, action_types):
        """Gets the action_size for the gym environment into the correct shape for a neural network"""
        if "action_size" in field.__dict__: return field.action_size
        if action_types == ActionType.DISCRETE:
            return field.action_space.n
        else:
            return field.action_space.shape[0]

    def get_action_shape(self, field):
        return field.action_space.shape
