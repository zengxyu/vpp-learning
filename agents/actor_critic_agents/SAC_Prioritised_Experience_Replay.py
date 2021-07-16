from agents.Base_Agent_AC import Base_Agent_AC
from memory.replay_buffer import ReplayBuffer, PriorityReplayBuffer
from utilities.OU_Noise import OU_Noise
from torch.optim import Adam
import torch
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6


class SAC_Prioritised_Experience_Replay(Base_Agent_AC):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
      https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
      to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SAC"

    def __init__(self, config):
        Base_Agent_AC.__init__(self, config)
        self.state_size = config.environment['state_size']
        self.action_size = config.environment['action_size']
        self.action_shape = config.environment['action_shape']
        # self.action_space = config.environment['action_space']

        self.critic_local = self.create_critic_network(state_dim=self.state_size,
                                                       action_dim=self.action_size, output_dim=1)
        self.critic_local_2 = self.create_critic_network(state_dim=self.state_size,
                                                         action_dim=self.action_size, output_dim=1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_target = self.create_critic_network(state_dim=self.state_size,
                                                        action_dim=self.action_size, output_dim=1)
        self.critic_target_2 = self.create_critic_network(state_dim=self.state_size,
                                                          action_dim=self.action_size, output_dim=1)
        Base_Agent_AC.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent_AC.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = PriorityReplayBuffer(buffer_size=self.hyperparameters['buffer_size'],
                                           batch_size=self.hyperparameters['batch_size'],
                                           device=self.device, seed=self.seed, is_discrete=False)
        self.actor_local = self.create_actor_network(state_dim=self.state_size,
                                                     action_dim=self.action_size,
                                                     output_dim=self.action_size * 2)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_shape).to(
                self.device)).item()  # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

    def create_actor_network(self, state_dim, action_dim, output_dim):
        ActorNetwork = self.config.actor_network
        net = ActorNetwork(state_dim=state_dim, action_dim=action_dim)
        return net

    def create_critic_network(self, state_dim, action_dim, output_dim):
        CriticNetwork = self.config.critic_network
        net = CriticNetwork(state_dim=state_dim, action_dim=action_dim)
        return net

    def pick_action(self, state, eval_ep=False):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        if isinstance(state, list):
            frame, robot_pose = state
            state = [torch.Tensor([frame]).to(self.device), torch.Tensor([robot_pose]).to(self.device)]
        else:
            state = torch.FloatTensor([state]).to(self.device)

        if eval_ep:
            action = self.actor_pick_action(state=state, eval=True)
        elif self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            action = np.random.rand(self.action_size) * 2 - 1
        else:
            action = self.actor_pick_action(state=state)
        if self.add_extra_noise:
            action += self.noise.sample()
        # print(action)
        return action

    def calculate_epsilon_with_exponent_strategy(self, epsilon, episode_number, epsilon_decay, epsilon_min):
        """Calculate epsilon according to an exponent of episode_number strategy"""
        epsilon = max(epsilon * epsilon_decay ** episode_number, epsilon_min)
        return epsilon

    def actor_pick_action(self, state=None, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""

        if eval == False:
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        # print("produce action state size:{}".format(state.shape))
        mean, log_std = self.actor_local(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # rsample means it is sampled using reparameterisation trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
               self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters[
                   "update_every_n_steps"] == 0

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        tree_idx, minibatch, ISWeights = self.memory.sample(is_vpp=self.config.environment['is_vpp'])
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = minibatch
        qf1_loss, td1_error, qf2_loss, td2_error = self.calculate_critic_losses(state_batch, action_batch,
                                                                                reward_batch, next_state_batch,
                                                                                mask_batch, ISWeights)
        td_errors = torch.abs(td1_error) + torch.abs(td2_error)

        self.update_critic_parameters(qf1_loss, qf2_loss)
        self.update_memory_batch_errors(tree_idx, td_errors, reward_batch)

        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        if self.automatic_entropy_tuning:
            alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        else:
            alpha_loss = None
        self.update_actor_parameters(policy_loss, alpha_loss)

        return qf1_loss.detach().cpu().numpy(), qf2_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, ISWeights):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(next_state_batch, next_state_action)
            qf2_next_target = self.critic_target_2(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (
                min_qf_next_target)
        qf1 = self.critic_local(state_batch, action_batch)
        qf2 = self.critic_local_2(state_batch, action_batch)
        qf1_loss = self.weighted_mse_loss(qf1, next_q_value, ISWeights)
        qf2_loss = self.weighted_mse_loss(qf2, next_q_value, ISWeights)

        td1_errors = qf1 - next_q_value
        td2_errors = qf2 - next_q_value

        return qf1_loss, td1_errors, qf2_loss, td2_errors

    def update_critic_parameters(self, critic_loss_1, critic_loss_2):
        """Updates the parameters for both critics"""

        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def update_actor_parameters(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def update_memory_batch_errors(self, tree_idx, td_errors, rewards):
        loss_reward_each_item = td_errors + rewards
        loss_reward_each_item = loss_reward_each_item.detach().cpu().numpy()
        tree_idx = tree_idx[:, np.newaxis]

        self.memory.batch_update(tree_idx, loss_reward_each_item)

    def weighted_mse_loss(self, input, target, weight):
        return torch.sum(weight * (input - target) ** 2)

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch, action)
        qf2_pi = self.critic_local_2(state_batch, action)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss
