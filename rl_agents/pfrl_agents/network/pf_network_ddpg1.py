import torch
from pfrl.nn import BoundByTanh
from pfrl.policies import DeterministicHead
from torch import nn
import torch.nn.functional as F


# q_func = nn.Sequential(
#     ConcatObsAndAction(),
#     nn.Linear(obs_size + action_size, 400),
#     nn.ReLU(),
#     nn.Linear(400, 300),
#     nn.ReLU(),
#     nn.Linear(300, 1),
# )
# policy = nn.Sequential(
#     nn.Linear(obs_size, 400),
#     nn.ReLU(),
#     nn.Linear(400, 300),
#     nn.ReLU(),
#     nn.Linear(300, action_size),
#     BoundByTanh(low=action_space.low, high=action_space.high),
#     DeterministicHead(),
# )


class QNetwork(nn.Module):
    def __init__(self, action_space, edge=3e-3):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(17 + action_space.action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)

    def forward(self, state, action):
        x = torch.cat((state, action), 1)

        x = F.relu(self.linear1(x))

        x = F.relu(self.linear2(x))

        x = self.linear3(x)

        return x


class PolicyNet(nn.Module):
    def __init__(self, action_space):
        super(PolicyNet, self).__init__()

        self.linear1 = nn.Linear(17, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, action_space.action_dim)
        self.bound = BoundByTanh(low=action_space.low, high=action_space.high),
        self.head = DeterministicHead()

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.bound(x)
        x = self.head(x)
        return x
