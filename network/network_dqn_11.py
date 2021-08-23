import pfrl
import torch
from torch import nn
import torch.nn.functional as F


class DQN_Network11_Improved(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(30, 48, kernel_size=4, stride=2)
        self.frame_con2 = torch.nn.Conv2d(48, 24, kernel_size=3, stride=1)
        # 2160
        self.frame_fc1 = torch.nn.Linear(2160, 512)
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
        out_frame = F.relu(self.frame_con2(out_frame))

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


class DQN_Network11(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
        self.frame_fc1 = torch.nn.Linear(3888, 512)
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
        # print(out_frame.size())
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


class DQN_Network11_Time_LSTM(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1_list = []
        self.frame_fc1_list = []
        self.frame_fc2_list = []

        self.pose_fc1a_list = []
        self.pose_fc2a_list = []

        self.pose_fc1b_list = []
        self.pose_fc2b_list = []

        for i in range(5):
            self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
            self.frame_fc1 = torch.nn.Linear(3888, 512)
            self.frame_fc2 = torch.nn.Linear(512, 128)

            self.pose_fc1a = torch.nn.Linear(3, 32)
            self.pose_fc2a = torch.nn.Linear(32, 64)

            self.pose_fc1b = torch.nn.Linear(3, 32)
            self.pose_fc2b = torch.nn.Linear(32, 64)

            self.frame_con1_list.append(self.frame_con1)
            self.frame_fc1_list.append(self.frame_fc1)
            self.frame_fc2_list.append(self.frame_fc2)
            self.pose_fc1a_list.append(self.pose_fc1a)
            self.pose_fc2a_list.append(self.pose_fc2a)
            self.pose_fc1b_list.append(self.pose_fc1b)
            self.pose_fc2b_list.append(self.pose_fc2b)

        self.pose_fc3 = torch.nn.Linear(256 * 5, 256 * 3)

        self.pose_fc4 = torch.nn.Linear(256 * 3, 256)

        self.pose_fc5 = torch.nn.Linear(256, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        frame, robot_pose = state
        frame_reshape = torch.transpose(frame, 0, 1)
        robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
        outs = None
        for i in range(5):
            out_frame = F.relu(self.frame_con1_list[i](frame_reshape[i]))
            # out_frame = F.relu(self.frame_con2(out_frame))
            # print(out_frame.size())
            out_frame = out_frame.reshape(out_frame.size()[0], -1)
            # print("out frame shape:", out_frame.shape)
            out_frame = F.relu(self.frame_fc1_list[i](out_frame))
            out_frame = F.relu(self.frame_fc2_list[i](out_frame))

            # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
            out_pose_a = F.relu(self.pose_fc1a_list[i](robot_pose_reshape[i][:, 0:3]))
            out_pose_a = F.relu(self.pose_fc2a_list[i](out_pose_a))

            out_pose_b = F.relu(self.pose_fc1b_list[i](robot_pose_reshape[i][:, 3:6]))
            out_pose_b = F.relu(self.pose_fc2b_list[i](out_pose_b))

            if outs is None:
                outs = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1)
            else:
                outs = torch.cat((outs, out_frame, out_pose_a, out_pose_b), dim=1)

        h0 = torch.zeros(1, out_frames.shape[0], self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        c0 = torch.zeros(1, out_frames.shape[0], self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        lstm_neighbor_output, (hn, cn) = self.lstm_neighbor(out_frames, (h0, c0))
        hn = hn.squeeze(0)
        out = torch.cat((hn, out_pose_a, out_pose_b), dim=1)

        out = F.relu(self.pose_fc3(outs))

        out = F.relu(self.pose_fc4(out))
        out = F.relu(self.pose_fc5(out))

        val = self.fc_val(out)
        return val


class DQN_Network11_Time(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1_list = []
        self.frame_fc1_list = []
        self.frame_fc2_list = []

        self.pose_fc1a_list = []
        self.pose_fc2a_list = []

        self.pose_fc1b_list = []
        self.pose_fc2b_list = []

        for i in range(5):
            self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
            self.frame_fc1 = torch.nn.Linear(3888, 512)
            self.frame_fc2 = torch.nn.Linear(512, 128)

            self.pose_fc1a = torch.nn.Linear(3, 32)
            self.pose_fc2a = torch.nn.Linear(32, 64)

            self.pose_fc1b = torch.nn.Linear(3, 32)
            self.pose_fc2b = torch.nn.Linear(32, 64)

            self.frame_con1_list.append(self.frame_con1)
            self.frame_fc1_list.append(self.frame_fc1)
            self.frame_fc2_list.append(self.frame_fc2)
            self.pose_fc1a_list.append(self.pose_fc1a)
            self.pose_fc2a_list.append(self.pose_fc2a)
            self.pose_fc1b_list.append(self.pose_fc1b)
            self.pose_fc2b_list.append(self.pose_fc2b)

        self.pose_fc3 = torch.nn.Linear(256 * 5, 256 * 3)

        self.pose_fc4 = torch.nn.Linear(256 * 3, 256)

        self.pose_fc5 = torch.nn.Linear(256, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        frame, robot_pose = state
        frame_reshape = torch.transpose(frame, 0, 1)
        robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
        outs = None
        for i in range(5):
            out_frame = F.relu(self.frame_con1_list[i](frame_reshape[i]))
            # out_frame = F.relu(self.frame_con2(out_frame))
            # print(out_frame.size())
            out_frame = out_frame.reshape(out_frame.size()[0], -1)
            # print("out frame shape:", out_frame.shape)
            out_frame = F.relu(self.frame_fc1_list[i](out_frame))
            out_frame = F.relu(self.frame_fc2_list[i](out_frame))

            # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
            out_pose_a = F.relu(self.pose_fc1a_list[i](robot_pose_reshape[i][:, 0:3]))
            out_pose_a = F.relu(self.pose_fc2a_list[i](out_pose_a))

            out_pose_b = F.relu(self.pose_fc1b_list[i](robot_pose_reshape[i][:, 3:6]))
            out_pose_b = F.relu(self.pose_fc2b_list[i](out_pose_b))

            if outs is None:
                outs = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1)
            else:
                outs = torch.cat((outs, out_frame, out_pose_a, out_pose_b), dim=1)
        out = F.relu(self.pose_fc3(outs))

        out = F.relu(self.pose_fc4(out))
        out = F.relu(self.pose_fc5(out))

        val = self.fc_val(out)
        return val


class DQN_Network11_Time_Deeper(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1_list = []
        self.frame_con2_list = []

        self.frame_fc1_list = []
        self.frame_fc2_list = []

        self.pose_fc1a_list = []
        self.pose_fc2a_list = []

        self.pose_fc1b_list = []
        self.pose_fc2b_list = []

        for i in range(5):
            frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
            frame_con2 = torch.nn.Conv2d(24, 48, kernel_size=4, stride=2, padding=1)

            frame_fc1 = torch.nn.Linear(1728, 512)
            frame_fc2 = torch.nn.Linear(512, 128)

            pose_fc1a = torch.nn.Linear(3, 32)
            pose_fc2a = torch.nn.Linear(32, 64)

            pose_fc1b = torch.nn.Linear(3, 32)
            pose_fc2b = torch.nn.Linear(32, 64)

            self.frame_con1_list.append(frame_con1)
            self.frame_con2_list.append(frame_con2)
            self.frame_fc1_list.append(frame_fc1)
            self.frame_fc2_list.append(frame_fc2)
            self.pose_fc1a_list.append(pose_fc1a)
            self.pose_fc2a_list.append(pose_fc2a)
            self.pose_fc1b_list.append(pose_fc1b)
            self.pose_fc2b_list.append(pose_fc2b)

        self.pose_fc3 = torch.nn.Linear(256 * 5, 256 * 3)

        self.pose_fc4 = torch.nn.Linear(256 * 3, 256)

        self.pose_fc5 = torch.nn.Linear(256, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        frame, robot_pose = state
        frame_reshape = torch.transpose(frame, 0, 1)
        robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
        outs = None
        for i in range(5):
            out_frame = F.relu(self.frame_con1_list[i](frame_reshape[i]))
            out_frame = F.relu(self.frame_con2_list[i](out_frame))

            out_frame = out_frame.reshape(out_frame.size()[0], -1)

            out_frame = F.relu(self.frame_fc1_list[i](out_frame))
            out_frame = F.relu(self.frame_fc2_list[i](out_frame))

            out_pose_a = F.relu(self.pose_fc1a_list[i](robot_pose_reshape[i][:, 0:3]))
            out_pose_a = F.relu(self.pose_fc2a_list[i](out_pose_a))

            out_pose_b = F.relu(self.pose_fc1b_list[i](robot_pose_reshape[i][:, 3:6]))
            out_pose_b = F.relu(self.pose_fc2b_list[i](out_pose_b))

            if outs is None:
                outs = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1)
            else:
                outs = torch.cat((outs, out_frame, out_pose_a, out_pose_b), dim=1)
        out = F.relu(self.pose_fc3(outs))

        out = F.relu(self.pose_fc4(out))
        out = F.relu(self.pose_fc5(out))

        val = self.fc_val(out)
        return val


class DQN_Network11_LSTM(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1s = []
        self.frame_fc1s = []
        self.frame_fc2s = []
        for i in range(5):
            self.frame_con1 = torch.nn.Conv2d(3, 12, kernel_size=4, stride=2, padding=1)
            self.frame_fc1 = torch.nn.Linear(1944, 512)
            self.frame_fc2 = torch.nn.Linear(512, 128)
            self.frame_con1s.append(self.frame_con1)
            self.frame_fc1s.append(self.frame_fc1)
            self.frame_fc2s.append(self.frame_fc2)

        self.hn_neighbor_state_dim = 512
        self.lstm_neighbor = nn.LSTM(128, self.hn_neighbor_state_dim, batch_first=True)
        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(128 + self.hn_neighbor_state_dim, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        frame, robot_pose = state
        frame = torch.reshape(frame, shape=(-1, 3, 5, frame.shape[-2], frame.shape[-1]))
        frame = torch.transpose(frame, 1, 2)
        out_frames = []
        for i in range(5):
            out_frame = F.relu(self.frame_con1s[i](frame[:, i]))
            # out_frame = F.relu(self.frame_con2(out_frame))
            # print(out_frame.size())
            out_frame = out_frame.reshape(out_frame.size()[0], -1)
            # print("out frame shape:", out_frame.shape)
            out_frame = F.relu(self.frame_fc1s[i](out_frame))
            out_frame = F.relu(self.frame_fc2s[i](out_frame))
            out_frames.append(out_frame.unsqueeze(1))
        out_frames = torch.cat(out_frames, dim=1)
        h0 = torch.zeros(1, out_frames.shape[0], self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        c0 = torch.zeros(1, out_frames.shape[0], self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        lstm_neighbor_output, (hn, cn) = self.lstm_neighbor(out_frames, (h0, c0))
        hn = hn.squeeze(0)

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out = torch.cat((hn, out_pose_a, out_pose_b), dim=1)
        # print(out.shape)
        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.fc_val(out)
        return val


class DQN_Network11_Without_RobotPose(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
        self.frame_fc1 = torch.nn.Linear(3888, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        # self.pose_fc1a = torch.nn.Linear(3, 32)
        # self.pose_fc2a = torch.nn.Linear(32, 64)
        #
        # self.pose_fc1b = torch.nn.Linear(3, 32)
        # self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(128, 64)

        self.pose_fc4 = torch.nn.Linear(64, 32)

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
        # out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        # out_pose_a = F.relu(self.pose_fc2a(out_pose_a))
        #
        # out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        # out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        # out = torch.cat((out_frame), dim=1)
        # print(out.shape)
        out = F.relu(self.pose_fc3(out_frame))

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
    def __init__(self, obs_size, action_size, n_atoms):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(3264, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256, 128)

        # self.pose_fc4 = torch.nn.Linear(128, 32)
        self.n_atoms = n_atoms
        self.action_size = action_size
        self.fc_val = torch.nn.Linear(128, self.action_size * self.n_atoms)
        self.register_buffer(
            "z_values", torch.linspace(0, 10000, self.n_atoms, dtype=torch.float)
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

        # out = F.relu(self.pose_fc4(out))

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
