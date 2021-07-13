from utilities.OU_Noise import OU_Noise
from exploration_strategies.Base_Exploration_Strategy import Base_Exploration_Strategy


class OU_Noise_Exploration(Base_Exploration_Strategy):
    """Ornstein-Uhlenbeck noise process exploration strategy"""

    def __init__(self, hyperparameters, action_size, seed):
        super().__init__(hyperparameters)
        self.noise = OU_Noise(action_size, seed, self.hyperparameters["mu"],
                              self.hyperparameters["theta"], self.hyperparameters["sigma"])

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action = action_info["action"]
        action += self.noise.sample()
        return action

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        raise ValueError("Must be implemented")

    def reset(self):
        """Resets the noise process"""
        self.noise.reset()
