from agents.Base_Agent_AC import Base_Agent_AC
from memory.replay_buffer import ReplayBuffer
from utilities.OU_Noise import OU_Noise
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6


class SAC(Base_Agent_AC):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
      https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
      to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SAC"

    def __init__(self, config):
        Base_Agent_AC.__init__(self, config)
        # assert self.action_types == "CONTINUOUS", "Action types must be continuous. Use SAC Discrete instead for discrete actions"
        # assert self.config.hyperparameters["Actor"][
        #            "final_layer_activation"] != "Softmax", "Final actor layer must not be softmax"
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
        self.memory = ReplayBuffer(buffer_size=self.hyperparameters['buffer_size'],
                                   batch_size=self.hyperparameters['batch_size'],
                                   device=self.device, seed=self.seed)
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

    def step(self, state, action, reward, next_state, done):
        self.memory.add_experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.global_step_number += 1

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
            action = np.rnn_andom.rand(self.action_size) * 2 - 1
            # action = self.action_space.sample()
            # print("Picking random action ", action)
        else:
            action = self.actor_pick_action(state=state)
        if self.add_extra_noise:
            action += self.noise.sample()
        # print(action)
        return action

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
        # print("action size:{}".format(self.action_size))
        # mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
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
        if self.enough_experiences_to_learn_from():
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()

            qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch,
                                                              mask_batch)
            self.update_critic_parameters(qf1_loss, qf2_loss)

            policy_loss, log_pi = self.calculate_actor_loss(state_batch)
            if self.automatic_entropy_tuning:
                alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
            else:
                alpha_loss = None
            self.update_actor_parameters(policy_loss, alpha_loss)

            return qf1_loss.detach().cpu().numpy(), qf2_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()
        return 0, 0, 0

    def sample_experiences(self):
        return self.memory.sample()

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(torch.cat((next_state_batch, next_state_action), 1))
            qf2_next_target = self.critic_target_2(torch.cat((next_state_batch, next_state_action), 1))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (
                min_qf_next_target)
        qf1 = self.critic_local(torch.cat((state_batch, action_batch), 1))
        qf2 = self.critic_local_2(torch.cat((state_batch, action_batch), 1))
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(torch.cat((state_batch, action), 1))
        qf2_pi = self.critic_local_2(torch.cat((state_batch, action), 1))
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

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

    def print_summary_of_latest_evaluation_episode(self):
        """Prints a summary of the latest episode"""
        print(" ")
        print("----------------------------")
        print("Episode score {} ".format(self.total_episode_score_so_far))
        print("----------------------------")
