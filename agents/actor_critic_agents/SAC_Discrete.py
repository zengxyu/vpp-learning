import torch
from torch.distributions import Categorical
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

from agents.Base_Agent_AC import Base_Agent_AC
from memory.replay_buffer import ReplayBuffer
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from agents.actor_critic_agents.SAC import SAC


class SAC_Discrete(SAC):
    """The Soft Actor Critic for discrete actions. It inherits from SAC for continuous actions and only changes a few
    methods."""
    agent_name = "SAC"

    def __init__(self, config):
        Base_Agent_AC.__init__(self, config)
        # assert self.action_types == "DISCRETE", "Action types must be discrete. Use SAC instead for continuous actions"
        # assert self.config.hyperparameters["Actor"]["final_layer_activation"] == "Softmax", "Final actor layer must be softmax"
        self.state_size = config.environment['state_size']
        self.action_size = config.environment['action_size']
        self.action_shape = config.environment['action_shape']
        self.action_space = config.environment['action_space']
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
        self.memory = ReplayBuffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"],
                                   self.device, self.config.seed)

        self.actor_local = self.create_actor_network(state_dim=self.state_size,
                                                     action_dim=self.action_size,
                                                     output_dim=self.action_size)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]
        assert not self.hyperparameters[
            "add_extra_noise"], "There is no add extra noise option for the discrete version of SAC at moment"
        self.add_extra_noise = False
        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

    def pick_action(self, state, eval_ep=False):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        if isinstance(state, list):
            frame, robot_pose = state
            state = [torch.Tensor([frame]).to(self.device), torch.Tensor([robot_pose]).to(self.device)]
        else:
            state = torch.FloatTensor([state]).to(self.device)

        if self.global_step_number == self.hyperparameters["min_steps_before_learning"]:
            print("=================================================================================")

        if eval_ep:
            action = self.actor_pick_action(state=state, eval=True)
        elif self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            # action = np.random.rand(self.action_size) * 2 - 1
            action = np.random.randint(0, self.action_size)
            # action = self.action_space.sample()
        else:
            action = self.actor_pick_action(state=state)
        if self.add_extra_noise:
            action += self.noise.sample()
        # print(action)
        return action

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        action_probabilities = self.actor_local(state)
        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        assert action_probabilities.size()[1] == self.action_size, "Actor output the wrong size"
        action_distribution = Categorical(action_probabilities)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, (
                action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(
                next_state_batch)
            qf1_next_target = self.critic_target(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probabilities * (
                    torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (
                min_qf_next_target)

        qf1 = self.critic_local(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities
