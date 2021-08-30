import torch
import torch.nn.functional as F


class DQN_Network11_Temporal(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
        self.frame_fc1 = torch.nn.Linear(3888, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256 * 6, 256 * 3)

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
            out_frame = F.relu(self.frame_con1(frame_reshape[i]))
            out_frame = out_frame.reshape(out_frame.size()[0], -1)
            out_frame = F.relu(self.frame_fc1(out_frame))
            out_frame = F.relu(self.frame_fc2(out_frame))

            # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
            out_pose_a = F.relu(self.pose_fc1a(robot_pose_reshape[i][:, 0:3]))
            out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

            out_pose_b = F.relu(self.pose_fc1b(robot_pose_reshape[i][:, 3:6]))
            out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

            if outs is None:
                outs = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1)
            else:
                outs = torch.cat((outs, out_frame, out_pose_a, out_pose_b), dim=1)

        out_frame = F.relu(self.frame_con1(frame_reshape[-1]))
        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        out_pose_a = F.relu(self.pose_fc1a(robot_pose_reshape[-1][:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose_reshape[-1][:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        outs = torch.cat((outs, out_frame, out_pose_a, out_pose_b), dim=1)

        out = F.relu(self.pose_fc3(outs))

        out = F.relu(self.pose_fc4(out))
        out = F.relu(self.pose_fc5(out))

        val = self.fc_val(out)
        return val


class DQN_Network11_Temporal_LSTM(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
        self.frame_fc1 = torch.nn.Linear(3888, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.hn_neighbor_state_dim = 512
        self.lstm_neighbor1 = torch.nn.LSTM(256, self.hn_neighbor_state_dim, batch_first=True)

        # self.pose_fc3 = torch.nn.Linear(512 + 256, 256 * 3)

        self.pose_fc4 = torch.nn.Linear(self.hn_neighbor_state_dim + 256, 256)

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
        batch_size = frame_reshape.shape[1]
        h0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        c0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        for i in range(5):
            out_frame = F.relu(self.frame_con1(frame_reshape[i]))
            out_frame = out_frame.reshape(out_frame.size()[0], -1)
            out_frame = F.relu(self.frame_fc1(out_frame))
            out_frame = F.relu(self.frame_fc2(out_frame))

            # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
            out_pose_a = F.relu(self.pose_fc1a(robot_pose_reshape[i][:, 0:3]))
            out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

            out_pose_b = F.relu(self.pose_fc1b(robot_pose_reshape[i][:, 3:6]))
            out_pose_b = F.relu(self.pose_fc2b(out_pose_b))
            out = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1).unsqueeze(1)
            if outs is None:
                outs = out
            else:
                outs = torch.cat((outs, out), dim=1)
        lstm_neighbor_output, (hn_frame, cn) = self.lstm_neighbor1(outs, (h0, c0))
        hn_frame = hn_frame.squeeze(0)

        out_frame = F.relu(self.frame_con1(frame_reshape[-1]))
        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        out_pose_a = F.relu(self.pose_fc1a(robot_pose_reshape[-1][:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose_reshape[-1][:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        outs = torch.cat((hn_frame, out_frame, out_pose_a, out_pose_b), dim=1)

        # out = F.relu(self.pose_fc3(outs))

        out = F.relu(self.pose_fc4(outs))
        out = F.relu(self.pose_fc5(out))

        val = self.fc_val(out)
        return val


class DQN_Network11_Temporal_LSTM2(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
        self.frame_fc1 = torch.nn.Linear(3888, 784)
        self.frame_fc2 = torch.nn.Linear(784, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.hn_neighbor_state_dim = 512
        self.lstm = torch.nn.LSTM(256, self.hn_neighbor_state_dim, batch_first=True)
        self.lstm_fc = torch.nn.Linear(self.hn_neighbor_state_dim, 256)

        self.pose_fc3 = torch.nn.Linear(256 + 256, 256)
        # self.pose_fc4 = torch.nn.Linear(512 + 256, 256)

        self.fc_val = torch.nn.Linear(256, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def encode(self, frame, robot_pose):
        out_frame = F.relu(self.frame_con1(frame))
        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1)
        return out

    def forward(self, state):
        frame, robot_pose = state
        frame_reshape = torch.transpose(frame, 0, 1)
        robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
        outs = None
        batch_size = frame_reshape.shape[1]
        h0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        c0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        for i in range(5):
            out = self.encode(frame_reshape[i], robot_pose_reshape[i])
            out = out.unsqueeze(1)
            if outs is None:
                outs = out
            else:
                outs = torch.cat((outs, out), dim=1)
        lstm_neighbor_output, (hn, cn) = self.lstm(outs, (h0, c0))
        hn = hn.squeeze(0)
        hn = F.relu(self.lstm_fc(hn))

        out = self.encode(frame_reshape[-1], robot_pose_reshape[-1])

        outs = torch.cat((hn, out), dim=1)

        out = F.relu(self.pose_fc3(outs))

        val = self.fc_val(out)
        return val


class DQN_Network11_Temporal_LSTM2_KnownMap(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
        self.frame_fc1 = torch.nn.Linear(3888, 784)
        self.frame_fc2 = torch.nn.Linear(784, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.hn_neighbor_state_dim = 512
        self.lstm = torch.nn.LSTM(256, self.hn_neighbor_state_dim, batch_first=True)
        self.lstm_fc = torch.nn.Linear(self.hn_neighbor_state_dim, 256)

        self.known_map_conv1 = torch.nn.Conv2d(24, 40, kernel_size=4, stride=2, padding=1)
        self.known_map_fc1 = torch.nn.Linear(6480, 1024)
        # self.known_map_fc2 = torch.nn.Linear(3240, 1024)
        self.known_map_fc3 = torch.nn.Linear(1024, 256)

        self.pose_fc3 = torch.nn.Linear(256 * 3, 256)
        # self.pose_fc4 = torch.nn.Linear(512 + 256, 256)

        self.fc_val = torch.nn.Linear(256, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def encode(self, frame, robot_pose):
        out_frame = F.relu(self.frame_con1(frame))
        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1)
        return out

    def encode_global_map(self, known_map):
        out_known_map = F.relu(self.known_map_conv1(known_map))
        out_known_map = out_known_map.reshape(out_known_map.size()[0], -1)
        out_known_map = F.relu(self.known_map_fc1(out_known_map))
        # out_known_map = F.relu(self.known_map_fc2(out_known_map))
        out_known_map = F.relu(self.known_map_fc3(out_known_map))
        return out_known_map

    def forward(self, state):
        known_map, frame, robot_pose = state
        frame_reshape = torch.transpose(frame, 0, 1)
        robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
        outs = None
        batch_size = frame_reshape.shape[1]
        h0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        c0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        for i in range(5):
            out = self.encode(frame_reshape[i], robot_pose_reshape[i])
            out = out.unsqueeze(1)
            if outs is None:
                outs = out
            else:
                outs = torch.cat((outs, out), dim=1)
        lstm_neighbor_output, (hn, cn) = self.lstm(outs, (h0, c0))
        hn = hn.squeeze(0)
        hn = F.relu(self.lstm_fc(hn))

        out_frame = self.encode(frame_reshape[-1], robot_pose_reshape[-1])
        out_known_map = self.encode_global_map(known_map)
        outs = torch.cat((hn, out_frame, out_known_map), dim=1)

        out = F.relu(self.pose_fc3(outs))

        val = self.fc_val(out)
        return val


class DQN_Network11_Temporal_LSTM3(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
        self.frame_fc1 = torch.nn.Linear(3888, 784)
        self.frame_fc2 = torch.nn.Linear(784, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.hn_neighbor_state_dim = 256
        self.lstm = torch.nn.LSTM(128, self.hn_neighbor_state_dim, batch_first=True)
        self.lstm_fc = torch.nn.Linear(self.hn_neighbor_state_dim, 128)

        self.pose_fc3 = torch.nn.Linear(128 * 3, 256)

        self.fc_val = torch.nn.Linear(256, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def encode_pos(self, robot_pose):
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out = torch.cat((out_pose_a, out_pose_b), dim=1)
        return out

    def encode_observation_map(self, frame):
        out_frame = F.relu(self.frame_con1(frame))
        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        return out_frame

    def forward(self, state):
        frame, robot_pose = state
        robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
        outs = None
        batch_size = frame.shape[0]
        seq_len = robot_pose_reshape.shape[0]
        h0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        c0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        for i in range(seq_len):
            out = self.encode_pos(robot_pose_reshape[i])
            out = out.unsqueeze(1)
            if outs is None:
                outs = out
            else:
                outs = torch.cat((outs, out), dim=1)
        lstm_neighbor_output, (hn, cn) = self.lstm(outs, (h0, c0))
        hn = hn.squeeze(0)
        hn = F.relu(self.lstm_fc(hn))

        out_pose = self.encode_pos(robot_pose_reshape[-1])
        out_frame = self.encode_observation_map(frame)
        outs = torch.cat((hn, out_pose, out_frame), dim=1)

        out = F.relu(self.pose_fc3(outs))

        val = self.fc_val(out)
        return val
