import torch
from torch import nn
import torch.nn.functional as F


# Soft Q Net
class SAC_QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(SAC_QNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
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


class SAC_PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(SAC_PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, action_dim)
        self.mean_linear.weight.data.uniform_(-edge, edge)
        self.mean_linear.bias.data.uniform_(-edge, edge)

        self.log_std_linear = nn.Linear(256, action_dim)
        self.log_std_linear.weight.data.uniform_(-edge, edge)
        self.log_std_linear.bias.data.uniform_(-edge, edge)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std


class DDPG_QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(DDPG_QNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
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


class DDPG_PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(DDPG_PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)

        return mean


class TD3_QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(TD3_QNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
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


class TD3_PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(TD3_PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, action_dim)

        self.mean_linear.weight.data.uniform_(-edge, edge)
        self.mean_linear.bias.data.uniform_(-edge, edge)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)

        return mean


class SAC_QNetwork3(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(SAC_QNetwork3, self).__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc1c = torch.nn.Linear(3, 32)
        self.pose_fc2c = torch.nn.Linear(32, 64)

        self.pose_fc1d = torch.nn.Linear(3, 32)
        self.pose_fc2d = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(128 + 64 * 4, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.linear = torch.nn.Linear(32, action_dim)
        self.linear.weight.data.uniform_(-edge, edge)
        self.linear.bias.data.uniform_(-edge, edge)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state, action):
        # frame, robot_pose = state
        frame, robot_pose = state
        out_frame = F.relu(self.frame_con1(frame))
        # out_frame = F.relu(self.frame_con2(out_frame))

        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        # print("out frame shape:", out_frame.shape)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))
        out_pose_c = F.relu(self.pose_fc1c(action[:, 0:3]))
        out_pose_c = F.relu(self.pose_fc2c(out_pose_c))
        out_pose_d = F.relu(self.pose_fc1d(action[:, 3:6]))
        out_pose_d = F.relu(self.pose_fc2d(out_pose_d))

        out = torch.cat((out_frame, out_pose_a, out_pose_b, out_pose_c, out_pose_d), dim=1)
        # print(out.shape)
        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.linear(out)
        return val


class SAC_PolicyNet3(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(SAC_PolicyNet3, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256, 128)
        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.mean_linear = nn.Linear(32, action_dim)
        self.mean_linear.weight.data.uniform_(-edge, edge)
        self.mean_linear.bias.data.uniform_(-edge, edge)

        self.log_std_linear = nn.Linear(32, action_dim)
        self.log_std_linear.weight.data.uniform_(-edge, edge)
        self.log_std_linear.bias.data.uniform_(-edge, edge)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        frame, robot_pose = state

        out_frame = F.relu(self.frame_con1(frame))
        # out_frame = F.relu(self.frame_con2(out_frame))

        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        # print("out frame shape:", out_frame.shape)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1)
        # print(out.shape)
        out = F.relu(self.pose_fc3(out))
        out = F.relu(self.pose_fc4(out))

        mean = self.mean_linear(out)
        log_std = self.log_std_linear(out)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class SAC_PER_QNetwork3(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(SAC_PER_QNetwork3, self).__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc1c = torch.nn.Linear(3, 32)
        self.pose_fc2c = torch.nn.Linear(32, 64)

        self.pose_fc1d = torch.nn.Linear(3, 32)
        self.pose_fc2d = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(128 + 64 * 4, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.linear = torch.nn.Linear(32, 1)
        self.linear.weight.data.uniform_(-edge, edge)
        self.linear.bias.data.uniform_(-edge, edge)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state, action):
        # frame, robot_pose = state
        frame, robot_pose = state
        out_frame = F.relu(self.frame_con1(frame))
        # out_frame = F.relu(self.frame_con2(out_frame))

        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        # print("out frame shape:", out_frame.shape)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))
        out_pose_c = F.relu(self.pose_fc1c(action[:, 0:3]))
        out_pose_c = F.relu(self.pose_fc2c(out_pose_c))
        out_pose_d = F.relu(self.pose_fc1d(action[:, 3:6]))
        out_pose_d = F.relu(self.pose_fc2d(out_pose_d))

        out = torch.cat((out_frame, out_pose_a, out_pose_b, out_pose_c, out_pose_d), dim=1)
        # print(out.shape)
        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.linear(out)
        return val


class SAC_PER_PolicyNet3(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(SAC_PER_PolicyNet3, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256, 128)
        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.mean_linear = nn.Linear(32, action_dim)
        self.mean_linear.weight.data.uniform_(-edge, edge)
        self.mean_linear.bias.data.uniform_(-edge, edge)

        self.log_std_linear = nn.Linear(32, action_dim)
        self.log_std_linear.weight.data.uniform_(-edge, edge)
        self.log_std_linear.bias.data.uniform_(-edge, edge)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        frame, robot_pose = state

        out_frame = F.relu(self.frame_con1(frame))
        # out_frame = F.relu(self.frame_con2(out_frame))

        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        # print("out frame shape:", out_frame.shape)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1)
        # print(out.shape)
        out = F.relu(self.pose_fc3(out))
        out = F.relu(self.pose_fc4(out))

        mean = self.mean_linear(out)
        log_std = self.log_std_linear(out)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class DDPG_QNetwork3(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(DDPG_QNetwork3, self).__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc1c = torch.nn.Linear(3, 32)
        self.pose_fc2c = torch.nn.Linear(32, 64)

        self.pose_fc1d = torch.nn.Linear(3, 32)
        self.pose_fc2d = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(128 + 64 * 4, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.linear = torch.nn.Linear(32, action_dim)
        self.linear.weight.data.uniform_(-edge, edge)
        self.linear.bias.data.uniform_(-edge, edge)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state, action):
        # frame, robot_pose = state
        frame, robot_pose = state
        out_frame = F.relu(self.frame_con1(frame))
        # out_frame = F.relu(self.frame_con2(out_frame))

        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        # print("out frame shape:", out_frame.shape)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))
        out_pose_c = F.relu(self.pose_fc1c(action[:, 0:3]))
        out_pose_c = F.relu(self.pose_fc2c(out_pose_c))
        out_pose_d = F.relu(self.pose_fc1d(action[:, 3:6]))
        out_pose_d = F.relu(self.pose_fc2d(out_pose_d))

        out = torch.cat((out_frame, out_pose_a, out_pose_b, out_pose_c, out_pose_d), dim=1)
        # print(out.shape)
        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.linear(out)
        return val


class DDPG_PolicyNet3(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(DDPG_PolicyNet3, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256, 128)
        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.mean_linear = nn.Linear(32, action_dim)
        self.mean_linear.weight.data.uniform_(-edge, edge)
        self.mean_linear.bias.data.uniform_(-edge, edge)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        frame, robot_pose = state

        out_frame = F.relu(self.frame_con1(frame))
        # out_frame = F.relu(self.frame_con2(out_frame))

        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        # print("out frame shape:", out_frame.shape)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1)
        # print(out.shape)
        out = F.relu(self.pose_fc3(out))
        out = F.relu(self.pose_fc4(out))

        out = self.mean_linear(out)
        out = torch.tanh(out)

        return out


class TD3_QNetwork3(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(TD3_QNetwork3, self).__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc1c = torch.nn.Linear(3, 32)
        self.pose_fc2c = torch.nn.Linear(32, 64)

        self.pose_fc1d = torch.nn.Linear(3, 32)
        self.pose_fc2d = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(128 + 64 * 4, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.linear = torch.nn.Linear(32, action_dim)
        self.linear.weight.data.uniform_(-edge, edge)
        self.linear.bias.data.uniform_(-edge, edge)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state, action):
        # frame, robot_pose = state
        frame, robot_pose = state
        out_frame = F.relu(self.frame_con1(frame))
        # out_frame = F.relu(self.frame_con2(out_frame))

        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        # print("out frame shape:", out_frame.shape)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))
        out_pose_c = F.relu(self.pose_fc1c(action[:, 0:3]))
        out_pose_c = F.relu(self.pose_fc2c(out_pose_c))
        out_pose_d = F.relu(self.pose_fc1d(action[:, 3:6]))
        out_pose_d = F.relu(self.pose_fc2d(out_pose_d))

        out = torch.cat((out_frame, out_pose_a, out_pose_b, out_pose_c, out_pose_d), dim=1)
        # print(out.shape)
        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        out = self.linear(out)
        out = torch.tanh(out)

        return out


class TD3_PolicyNet3(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(TD3_PolicyNet3, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256, 128)
        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.mean_linear = nn.Linear(32, action_dim)
        self.mean_linear.weight.data.uniform_(-edge, edge)
        self.mean_linear.bias.data.uniform_(-edge, edge)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        frame, robot_pose = state

        out_frame = F.relu(self.frame_con1(frame))
        # out_frame = F.relu(self.frame_con2(out_frame))

        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        # print("out frame shape:", out_frame.shape)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1)
        # print(out.shape)
        out = F.relu(self.pose_fc3(out))
        out = F.relu(self.pose_fc4(out))

        out = self.mean_linear(out)
        out = torch.tanh(out)

        return out
