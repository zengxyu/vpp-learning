import torch
import torch.nn as nn
import torch.nn.functional as F


class PPO_LSTM(nn.Module):
    def __init__(self, action_size, num_layers):
        super().__init__()
        self.fc1 = torch.nn.Linear(324, 128)
        self.fc2 = torch.nn.Linear(128, 32)

        self.pose_fc1 = torch.nn.Linear(7, 32)

        # lstm
        self.lstm = torch.nn.LSTM(input_size=64, hidden_size=32, num_layers=num_layers)

        self.fc_val = torch.nn.Linear(32, 1)
        self.fc_pol = torch.nn.Linear(32, action_size)

    def forward(self, frame, robot_pose, h_in, c_in):
        frame_size = frame.size()
        pose_size = robot_pose.size()

        frame = frame.reshape(frame_size[0] * frame_size[1], -1)
        robot_pose = robot_pose.reshape(pose_size[0] * pose_size[1], -1)

        out1 = F.relu(self.fc1(frame))

        # out1 : [batch_size*seq_len, dim, h, w]
        out1 = F.relu(self.fc2(out1))

        # out2 : [batch_size*seq_len, input_size]
        out2 = F.relu(self.pose_fc1(robot_pose))

        # out : [batch_size*seq_len, input_size]
        out = torch.cat((out1, out2), dim=1)

        out = out.reshape(frame_size[0], frame_size[1], -1)

        out, (h_out, c_out) = self.lstm(out, (h_in, c_in))
        # ===================================================changed
        val = self.fc_val(out)
        pol = self.fc_pol(out)
        pol = F.softmax(pol, dim=2)
        return val, pol, h_out, c_out


class PPO_LSTM2(nn.Module):
    """
    with only robot pose passing to the lstm
    """

    def __init__(self, action_size, num_layers):
        super().__init__()
        # seq_len, batch size, depth, h ,w
        self.fc1 = torch.nn.Linear(324, 128)
        self.fc2 = torch.nn.Linear(128, 32)

        # lstm
        self.lstm = torch.nn.LSTM(input_size=7, hidden_size=32, num_layers=num_layers)

        self.fc_val = torch.nn.Linear(64, 1)
        self.fc_pol = torch.nn.Linear(64, action_size)

    def forward(self, frame, robot_pose, h_in, c_in):
        frame_size = frame.size()
        frame = frame.reshape(frame_size[0] * frame_size[1], -1)

        out1 = F.relu(self.fc1(frame))

        # out1 : [batch_size*seq_len, dim, h, w]
        out1 = F.relu(self.fc2(out1))

        # out1 : [batch_size, seq_len, input_size]
        out1 = out1.reshape(frame_size[0], frame_size[1], -1)

        out2, (h_out, c_out) = self.lstm(robot_pose, (h_in, c_in))
        # out : [batch_size*seq_len, input_size]
        out = torch.cat((out1, out2), dim=2)

        # ===================================================changed
        val = self.fc_val(out)
        pol = self.fc_pol(out)
        pol = F.softmax(pol, dim=2)
        return val, pol, h_out, c_out


class PPO_LSTM3(nn.Module):
    def __init__(self, action_size, num_layers):
        super().__init__()
        # seq_len, batch size, depth, h ,w
        self.frame_con1 = torch.nn.Conv2d(3, 16, kernel_size=4, stride=2)
        self.frame_con2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(72, 32)
        self.pose_fc1 = torch.nn.Linear(2, 32)
        # lstm
        self.lstm = torch.nn.LSTM(input_size=64, hidden_size=32, num_layers=num_layers)

        self.fc_val = torch.nn.Linear(32, 1)
        self.fc_pol = torch.nn.Linear(32, action_size)

    def forward(self, frame, robot_pose, h_in, c_in):
        frame_size = frame.size()
        pose_size = robot_pose.size()

        frame = frame.reshape(frame_size[0] * frame_size[1], frame_size[2], frame_size[3], frame_size[4])
        robot_pose = robot_pose.reshape(pose_size[0] * pose_size[1], pose_size[-1])

        out1 = F.relu(self.frame_con1(frame))

        # out1 : [batch_size*seq_len, dim, h, w]
        out1 = F.relu(self.frame_con2(out1))

        # out1 : [batch_size*seq_len, input_size]
        out1 = out1.reshape(out1.size()[0], -1)
        out1 = F.relu(self.frame_fc1(out1))

        # out2 : [batch_size*seq_len, input_size]
        out2 = F.relu(self.pose_fc1(robot_pose))

        # out : [batch_size*seq_len, input_size]
        out = torch.cat((out1, out2), dim=1)
        out_size = out.size()

        out = out.reshape(frame_size[0], frame_size[1], out_size[1])

        out, (h_out, c_out) = self.lstm(out, (h_in, c_in))
        # ===================================================changed
        val = self.fc_val(out)
        pol = self.fc_pol(out)
        pol = F.softmax(pol, dim=2)
        return val, pol, h_out, c_out


class PPO_LSTM4(nn.Module):
    """
    with only robot pose passing to the lstm
    """

    def __init__(self, action_size, num_layers):
        super().__init__()
        # seq_len, batch size, depth, h ,w
        self.frame_con1 = torch.nn.Conv2d(3, 16, kernel_size=4, stride=2)
        self.frame_con2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(288, 96)
        self.frame_fc2 = torch.nn.Linear(96, 32)

        # lstm
        self.lstm = torch.nn.LSTM(input_size=7, hidden_size=32, num_layers=num_layers)

        self.fc_val = torch.nn.Linear(64, 1)
        self.fc_pol = torch.nn.Linear(64, action_size)

    def forward(self, frame, robot_pose, h_in, c_in):
        frame_size = frame.size()

        frame = frame.reshape(frame_size[0] * frame_size[1], frame_size[2], frame_size[3], frame_size[4])
        # robot_pose = robot_pose.reshape(pose_size[0] * pose_size[1], pose_size[-1])

        out1 = F.relu(self.frame_con1(frame))

        # out1 : [batch_size*seq_len, dim, h, w]
        out1 = F.relu(self.frame_con2(out1))

        # out1 : [batch_size, seq_len, input_size]
        out1 = out1.reshape(frame_size[0], frame_size[1], -1)
        out1 = F.relu(self.frame_fc1(out1))
        out1 = F.relu(self.frame_fc2(out1))

        out2, (h_out, c_out) = self.lstm(robot_pose, (h_in, c_in))
        # out : [batch_size*seq_len, input_size]
        out = torch.cat((out1, out2), dim=2)

        # ===================================================changed
        val = self.fc_val(out)
        pol = self.fc_pol(out)
        pol = F.softmax(pol, dim=2)
        return val, pol, h_out, c_out


class PPO_LSTM5(nn.Module):
    """
    with only robot pose passing to the lstm
    """

    def __init__(self, action_size=13, robot_pose_size=7, num_layers=4):
        super().__init__()
        # seq_len, batch size, depth, h ,w
        self.frame_con1 = torch.nn.Conv2d(3, 16, kernel_size=4, stride=2)
        self.frame_con2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(288, 96)
        self.frame_fc2 = torch.nn.Linear(96, 32)

        # lstm
        self.lstm1 = torch.nn.LSTM(input_size=3, hidden_size=16, num_layers=num_layers)
        self.lstm2 = torch.nn.LSTM(input_size=7, hidden_size=16, num_layers=num_layers)

        self.fc_val = torch.nn.Linear(64, 1)
        self.fc_pol = torch.nn.Linear(64, action_size)

    def forward(self, frame, robot_pose, h_in, c_in):
        frame_size = frame.size()
        robot_pose_size = robot_pose.size()

        frame = frame.reshape(frame_size[0] * frame_size[1], frame_size[2], frame_size[3], frame_size[4])
        # robot_pose = robot_pose.reshape(pose_size[0] * pose_size[1], pose_size[-1])

        out1 = F.relu(self.frame_con1(frame))

        # out1 : [batch_size*seq_len, dim, h, w]
        out1 = F.relu(self.frame_con2(out1))

        # out1 : [batch_size, seq_len, input_size]
        out1 = out1.reshape(frame_size[0], frame_size[1], -1)
        out1 = F.relu(self.frame_fc1(out1))
        out1 = F.relu(self.frame_fc2(out1))

        h_in_size = h_in.size()
        half_size = int(h_in_size[-1] / 2)
        rp = robot_pose[:, :, :3]

        h_in1 = h_in[:, :, :half_size].contiguous()
        c_in1 = c_in[:, :, :half_size].contiguous()

        h_in2 = h_in[:, :, half_size:].contiguous()
        c_in2 = c_in[:, :, half_size:].contiguous()

        out21, (h_out1, c_out1) = self.lstm1(rp, (h_in1, c_in1))
        out22, (h_out2, c_out2) = self.lstm2(robot_pose, (h_in2, c_in2))
        h_out = torch.cat((h_out1, h_out2), dim=2)
        c_out = torch.cat((c_out1, c_out2), dim=2)

        # out : [batch_size*seq_len, input_size]
        out = torch.cat((out1, out21, out22), dim=2)

        # ===================================================changed
        val = self.fc_val(out)
        pol = self.fc_pol(out)
        pol = F.softmax(pol, dim=2)
        return val, pol, h_out, c_out
