import pfrl
import torch
from torch import nn
import torch.nn.functional as F


class DQN_Network_Shadow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.con1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=4)
        self.con1 = torch.nn.Conv2d(1, 4, kernel_size=3, stride=1)

        # self.con3 = torch.nn.Conv2d(32, 32, 3)
        # self.sm = torch.nn.Softmax2d()
        self.fc1 = torch.nn.Linear(5776, 2048)

        # self.fc1 = torch.nn.Linear(256, 32)
        self.fc2 = torch.nn.Linear(2048, 512)
        self.fc3 = torch.nn.Linear(512, 32)

        self.fc_pose = torch.nn.Linear(2, 32)

        self.fc_pol = torch.nn.Linear(64, 4)

        # Initialize neural network weights
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, frame, robot_pose):
        out = self.con1(frame)
        out = F.relu(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)

        out_pose = self.fc_pose(robot_pose)
        out_pose = F.relu(out_pose)
        out = torch.cat((out, out_pose), dim=1)
        # ===================================================changed
        q_value = self.fc_pol(out)
        return q_value


class DQN_Network4(torch.nn.Module):
    def __init__(self, action_space):
        super().__init__()

        self.pose_fc1 = torch.nn.Linear(7, 32)
        self.pose_fc2 = torch.nn.Linear(32, 128)

        self.concat_fc = torch.nn.Linear(128, 64)

        self.fc_val = torch.nn.Linear(64, action_space)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, robot_pose):

        out_pose = F.relu(self.pose_fc1(robot_pose))
        out_pose = F.relu(self.pose_fc2(out_pose))

        out = F.relu(self.concat_fc(out_pose))

        q_value = self.fc_val(out)
        return q_value


class DQN_Network5(torch.nn.Module):
    def __init__(self, action_size=13, robot_pose_size=7):
        super().__init__()

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc1c = torch.nn.Linear(4, 32)
        self.pose_fc2c = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(192, 96)

        self.fc_val = torch.nn.Linear(96, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, robot_pose):
        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, :3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out_pose_c = F.relu(self.pose_fc1c(robot_pose[:, 6:]))
        out_pose_c = F.relu(self.pose_fc2c(out_pose_c))

        out = torch.cat((out_pose_a, out_pose_b, out_pose_c), dim=1)

        out = F.relu(self.pose_fc3(out))

        val = self.fc_val(out)
        return val


class DQN_Network7(torch.nn.Module):
    def __init__(self, action_size=13, robot_pose_size=7):
        super().__init__()

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc1c = torch.nn.Linear(3, 32)
        self.pose_fc2c = torch.nn.Linear(32, 64)

        self.pose_fc1d = torch.nn.Linear(3, 32)
        self.pose_fc2d = torch.nn.Linear(32, 64)

        self.pose_fc1e = torch.nn.Linear(3, 32)
        self.pose_fc2e = torch.nn.Linear(32, 64)

        self.pose_fc1f = torch.nn.Linear(3, 32)
        self.pose_fc2f = torch.nn.Linear(32, 64)

        self.pose_fc1g = torch.nn.Linear(4, 32)
        self.pose_fc2g = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(448, 256)

        self.pose_fc4 = torch.nn.Linear(256, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, robot_pose):
        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out_pose_c = F.relu(self.pose_fc1c(robot_pose[:, 6:9]))
        out_pose_c = F.relu(self.pose_fc2c(out_pose_c))

        out_pose_d = F.relu(self.pose_fc1d(robot_pose[:, 9:12]))
        out_pose_d = F.relu(self.pose_fc2d(out_pose_d))

        out_pose_e = F.relu(self.pose_fc1e(robot_pose[:, 12:15]))
        out_pose_e = F.relu(self.pose_fc2e(out_pose_e))

        out_pose_f = F.relu(self.pose_fc1f(robot_pose[:, 15:18]))
        out_pose_f = F.relu(self.pose_fc2f(out_pose_f))

        out_pose_g = F.relu(self.pose_fc1g(robot_pose[:, 18:]))
        out_pose_g = F.relu(self.pose_fc2g(out_pose_g))

        out = torch.cat((out_pose_a, out_pose_b, out_pose_c, out_pose_d, out_pose_e, out_pose_f, out_pose_g), dim=1)

        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.fc_val(out)
        return val


class DQN_Network8(torch.nn.Module):
    def __init__(self, action_size=13, robot_pose_size=7):
        super().__init__()

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(128, 256)

        self.pose_fc4 = torch.nn.Linear(256, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, robot_pose):
        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out = torch.cat((out_pose_a, out_pose_b), dim=1)

        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.fc_val(out)
        return val


class DQN_Network9(torch.nn.Module):
    def __init__(self, action_size=13, robot_pose_size=7):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 32, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(2048, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, frame, robot_pose):

        out_frame = F.relu(self.frame_con1(frame))
        # out_frame = F.relu(self.frame_con2(out_frame))

        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1)

        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.fc_val(out)
        return val


class DQN_Network12(torch.nn.Module):
    def __init__(self, action_size=13, robot_pose_size=7):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(6, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(192, 108)

        self.pose_fc4 = torch.nn.Linear(108, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, frame, robot_pose):

        out_frame = F.relu(self.frame_con1(frame))
        # out_frame = F.relu(self.frame_con2(out_frame))

        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        # print("out frame shape:", out_frame.shape)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        # out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        # out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out = torch.cat((out_frame, out_pose_a), dim=1)
        # print(out.shape)
        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.fc_val(out)
        return val


class DQN_Network13(torch.nn.Module):
    def __init__(self, action_size=13, robot_pose_size=7):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc1c = torch.nn.Linear(6, 32)
        self.pose_fc2c = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(320, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, frame, robot_pose):

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

        out_pose_c = F.relu(self.pose_fc1c(robot_pose[:, 6:]))
        out_pose_c = F.relu(self.pose_fc2c(out_pose_c))

        out = torch.cat((out_frame, out_pose_a, out_pose_b, out_pose_c), dim=1)
        # print(out.shape)
        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.fc_val(out)
        return val


class DQN_Network_CartPole(nn.Module):
    def __init__(self, input_size=4, hidden=256, action_size=2):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(input_size, 256)
        self.l2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class DQN_Network_Dueling_CartPole(nn.Module):
    def __init__(self, input_size=4, hidden=256, action_size=2):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(input_size, hidden)
        self.advantage = nn.Linear(hidden, action_size)

        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - torch.mean(advantage, dim=1).unsqueeze(1)


class DQN_Network12(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(576, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

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

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1)
        # print(out.shape)
        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.fc_val(out)
        return val


class DQN_Network11(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

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

        val = self.fc_val(out)
        return val


class DQN_Network11_PFRL(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

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

        val = self.fc_val(out)
        return pfrl.action_value.DiscreteActionValue(val)


class DQN_Network11_PFRL_Rainbow(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)
        self.n_atoms = 60
        self.action_size = action_size
        self.fc_val = torch.nn.Linear(32, self.action_size * self.n_atoms)
        self.register_buffer(
            "z_values", torch.linspace(-100, 100, self.n_atoms, dtype=torch.float)
        )

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

        val = self.fc_val(out)
        v_logits = val.reshape((-1, self.action_size, self.n_atoms))
        probs = nn.functional.softmax(v_logits, dim=2)
        return pfrl.action_value.DistributionalDiscreteActionValue(probs, self.z_values)


class DQN_Network11_With_Multiplier(torch.nn.Module):
    def __init__(self, action_size, multiplier_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

        self.mt_val = torch.nn.Linear(action_size + 128, multiplier_size)

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

        fc_val_out = self.fc_val(out)

        mt_in = torch.cat((fc_val_out, out), dim=1)
        mt_val_out = self.mt_val(mt_in)

        return torch.cat((fc_val_out, mt_val_out), dim=1)


class DQN_Network11_Dueling(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        # self.fc_val = torch.nn.Linear(32, action_size)

        self.advantage = nn.Linear(32, action_size)

        self.value = nn.Linear(32, 1)

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

        advantage = self.advantage(out)
        value = self.value(out)

        return value + advantage - torch.mean(advantage, dim=1).unsqueeze(1)


class DQN_Network10(torch.nn.Module):
    def __init__(self, action_size=13, robot_pose_size=7):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 32, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(2048, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(4, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, frame, robot_pose):

        out_frame = F.relu(self.frame_con1(frame))
        # out_frame = F.relu(self.frame_con2(out_frame))

        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:7]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1)

        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.fc_val(out)
        return val


class DQN_Network6(torch.nn.Module):
    def __init__(self, action_size=13, robot_pose_size=7):
        super().__init__()

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc1c = torch.nn.Linear(3, 32)
        self.pose_fc2c = torch.nn.Linear(32, 64)

        self.pose_fc1d = torch.nn.Linear(4, 32)
        self.pose_fc2d = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, robot_pose):
        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out_pose_c = F.relu(self.pose_fc1c(robot_pose[:, 6:9]))
        out_pose_c = F.relu(self.pose_fc2c(out_pose_c))

        out_pose_d = F.relu(self.pose_fc1d(robot_pose[:, 9:]))
        out_pose_d = F.relu(self.pose_fc2d(out_pose_d))

        out = torch.cat((out_pose_a, out_pose_b, out_pose_c, out_pose_d), dim=1)

        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.fc_val(out)
        return val


class DQN_Network(torch.nn.Module):
    def __init__(self, action_space):
        super().__init__()

        self.frame_con1 = torch.nn.Conv2d(3, 16, kernel_size=4, stride=2)
        self.frame_con2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(288, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1 = torch.nn.Linear(7, 32)
        self.pose_fc2 = torch.nn.Linear(32, 128)

        self.concat_fc = torch.nn.Linear(256, 64)

        self.fc_val = torch.nn.Linear(64, action_space)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, frame, robot_pose):
        out_frame = F.relu(self.frame_con1(frame))
        out_frame = F.relu(self.frame_con2(out_frame))

        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        out_pose = F.relu(self.pose_fc1(robot_pose))
        out_pose = F.relu(self.pose_fc2(out_pose))

        out = torch.cat((out_frame, out_pose), dim=1)

        out = F.relu(self.concat_fc(out))

        q_value = self.fc_val(out)
        return q_value


class DQN_NetworkDeeper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.con1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=4)
        self.con1 = torch.nn.Conv2d(1, 16, kernel_size=8, stride=4)

        self.con2 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.con3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1)

        # self.con3 = torch.nn.Conv2d(32, 32, 3)
        # self.sm = torch.nn.Softmax2d()
        self.fc1 = torch.nn.Linear(64, 32)

        # self.fc1 = torch.nn.Linear(256, 32)
        # self.fc2 = torch.nn.Linear(64, 64)
        # self.fc3 = torch.nn.Linear(64, 64)

        self.fc_pose = torch.nn.Linear(2, 32)

        self.fc_pol1 = torch.nn.Linear(64, 32)
        self.fc_pol2 = torch.nn.Linear(32, 4)

        # Initialize neural network weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, frame, robot_pose):
        out = self.con1(frame)
        out = F.relu(out)
        out = self.con2(out)
        out = F.relu(out)
        out = self.con3(out)
        out = F.relu(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)

        out_pose = self.fc_pose(robot_pose)
        out_pose = F.relu(out_pose)
        out = torch.cat((out, out_pose), dim=1)
        # ===================================================changed
        out = self.fc_pol1(out)
        out = F.relu(out)
        q_value = self.fc_pol2(out)
        return q_value


class DQN_NetworkDeeper2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.con1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=4)
        self.con1 = torch.nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.con2 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.con3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1)

        self.con4 = torch.nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.con5 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.con6 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # self.con3 = torch.nn.Conv2d(32, 32, 3)
        # self.sm = torch.nn.Softmax2d()
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(64, 32)

        # self.fc1 = torch.nn.Linear(256, 32)
        # self.fc2 = torch.nn.Linear(64, 64)
        # self.fc3 = torch.nn.Linear(64, 64)

        self.fc_pose = torch.nn.Linear(2, 32)

        self.fc_pol1 = torch.nn.Linear(96, 48)
        self.fc_pol2 = torch.nn.Linear(48, 4)

        # Initialize neural network weights
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, frame, frame2, robot_pose):
        out = self.con1(frame)
        out = F.relu(out)
        out = self.con2(out)
        out = F.relu(out)
        out = self.con3(out)
        out = F.relu(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)

        out2 = self.con4(frame)
        out2 = F.relu(out2)
        out2 = self.con5(out2)
        out2 = F.relu(out2)
        out2 = self.con6(out2)
        out2 = F.relu(out2)
        out2 = out2.reshape(out2.size(0), -1)
        out2 = self.fc2(out2)
        out2 = F.relu(out2)

        out_pose = self.fc_pose(robot_pose)
        out_pose = F.relu(out_pose)

        out = torch.cat((out, out2, out_pose), dim=1)
        # ===================================================changed
        out = self.fc_pol1(out)

        q_value = self.fc_pol2(out)
        return q_value


class Linear_DQN_Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(1600, 784)
        self.fc2 = torch.nn.Linear(784, 256)
        self.fc3 = torch.nn.Linear(256, 32)

        self.fc_pose = torch.nn.Linear(2, 32)

        self.fc_pol = torch.nn.Linear(64, 4)

        # Initialize neural network weights
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, frame, robot_pose):
        frame = frame.reshape(frame.size(0), -1)
        robot_pose = robot_pose.reshape(robot_pose.size(0), -1)

        out = self.fc1(frame)
        out = F.relu(out)

        out = self.fc2(out)
        out = F.relu(out)

        out = self.fc3(out)
        out = F.relu(out)

        out_pose = self.fc_pose(robot_pose)
        out_pose = F.relu(out_pose)
        out = torch.cat((out, out_pose), dim=1)
        # ===================================================changed
        q_value = self.fc_pol(out)
        return q_value


class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        self.fc1 = torch.nn.Linear(1600, 784)
        self.fc2 = torch.nn.Linear(784, 256)
        self.fc3 = torch.nn.Linear(256, 32)

        self.fc_pose = torch.nn.Linear(2, 32)

        self.fc_value = torch.nn.Linear(64, 128)
        self.fc_adv = torch.nn.Linear(64, 128)

        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, 4)

    def forward(self, frame, robot_pose):
        frame = frame.reshape(frame.size(0), -1)
        out = self.fc1(frame)
        out = F.relu(out)

        out = self.fc2(out)
        out = F.relu(out)

        out = self.fc3(out)
        out = F.relu(out)

        out_pose = self.fc_pose(robot_pose)
        out_pose = F.relu(out_pose)
        out = torch.cat((out, out_pose), dim=1)

        out_value = F.relu(self.fc_value(out))
        out_adv = F.relu(self.fc_adv(out))

        value = self.value(out_value)
        adv = self.adv(out_adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q

    # def select_action(self, state):
    #     with torch.no_grad():
    #         Q = self.forward(state)
    #         action_index = torch.argmax(Q, dim=1)
    #     return action_index.item()
