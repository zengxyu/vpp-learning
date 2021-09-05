import torch
import torch.nn.functional as F
import numpy as np


class DQN_Network11_Explore(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        # 3 5 4 2
        # 15 36 18
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
        self.frame_fc1 = torch.nn.Linear(3888, 784)
        self.frame_fc2 = torch.nn.Linear(784, 128)

        self.frame_nb_fc1 = torch.nn.Linear(405, 128)
        self.frame_nb_fc2 = torch.nn.Linear(128, 48)

        self.hn_neighbor_state_dim = 192
        self.lstm_neighbor = torch.nn.LSTM(48, self.hn_neighbor_state_dim, batch_first=True)
        self.lstm_neighbor_fc = torch.nn.Linear(self.hn_neighbor_state_dim, 64)
        # lstm neighbor output = 128

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(128 + 64, 128)

        self.fc_val = torch.nn.Linear(128, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def encode_neighbor(self, frame):
        # 15 36 18
        batch_size, depth, w, h = frame.shape
        frame = torch.reshape(frame, (batch_size, 3, 5, 36, 18))
        known_frame = frame[:, 0, :, :, :]  # batch_size, 5, 36, 18
        known_frame = torch.reshape(known_frame, (-1, 5, 4, 9, 2, 9))
        known_frame = torch.transpose(known_frame, 1, 2)  # batch_size, 4, 5, 9, 2, 9
        known_frame = torch.transpose(known_frame, 3, 4)  # batch_size, 4, 5, 2, 9, 9
        known_frame = torch.transpose(known_frame, 2, 3)  # batch_size, 4, 2, 5, 9, 9
        known_frame = torch.reshape(known_frame, (batch_size, 8, 405))
        known_frame = torch.transpose(known_frame, 0, 1)  #
        seq_len = known_frame.shape[0]  # 8
        h0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        c0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        outs = None
        for i in range(seq_len):
            out = self.encode_region(known_frame[i])
            out = out.unsqueeze(1)
            if outs is None:
                outs = out
            else:
                outs = torch.cat((outs, out), dim=1)
        lstm_neighbor_output, (hn, cn) = self.lstm_neighbor(outs, (h0, c0))
        hn_neighbor = hn.squeeze(0)
        hn_neighbor = F.relu(self.lstm_neighbor_fc(hn_neighbor))
        return hn_neighbor

    def encode_region(self, region_frame):
        # out_frame = F.relu(self.frame_nb_con1(region_frame))
        # out_frame = F.relu(self.frame_nb_con2(out_frame))

        # out_frame = region_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_nb_fc1(region_frame))
        out_frame = F.relu(self.frame_nb_fc2(out_frame))

        return out_frame

    def encode_observation_map(self, frame):
        out_frame = F.relu(self.frame_con1(frame))
        # out_frame = F.relu(self.frame_con2(out_frame))
        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        return out_frame

    def forward(self, state):
        frame, robot_pose = state
        batch_size = robot_pose.shape[0]
        seq_len = robot_pose.shape[1]
        robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
        out_frame = self.encode_observation_map(frame)
        frame_spacial = self.encode_neighbor(frame)
        outs = torch.cat((out_frame, frame_spacial), dim=1)

        out = F.relu(self.pose_fc3(outs))

        val = self.fc_val(out)
        return val


class DQN_Network11_Exploit(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        # 3 5 4 2
        # 15 36 18
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
        self.frame_fc1 = torch.nn.Linear(3888, 784)
        self.frame_fc2 = torch.nn.Linear(784, 128)

        self.frame_nb_fc1 = torch.nn.Linear(405, 128)
        self.frame_nb_fc2 = torch.nn.Linear(128, 48)

        self.hn_neighbor_state_dim = 192
        self.lstm_neighbor = torch.nn.LSTM(48, self.hn_neighbor_state_dim, batch_first=True)
        self.lstm_neighbor_fc = torch.nn.Linear(self.hn_neighbor_state_dim, 64)
        # lstm neighbor output = 128

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(128 + 64, 128)

        self.fc_val = torch.nn.Linear(128, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def encode_neighbor(self, frame):
        # 15 36 18
        batch_size, depth, w, h = frame.shape
        frame = torch.reshape(frame, (batch_size, 3, 5, 36, 18))
        known_frame = frame[:, 2, :, :, :]  # batch_size, 5, 36, 18
        known_frame = torch.reshape(known_frame, (-1, 5, 4, 9, 2, 9))
        known_frame = torch.transpose(known_frame, 1, 2)  # batch_size, 4, 5, 9, 2, 9
        known_frame = torch.transpose(known_frame, 3, 4)  # batch_size, 4, 5, 2, 9, 9
        known_frame = torch.transpose(known_frame, 2, 3)  # batch_size, 4, 2, 5, 9, 9
        known_frame = torch.reshape(known_frame, (batch_size, 8, 405))
        known_frame = torch.transpose(known_frame, 0, 1)  #
        seq_len = known_frame.shape[0]  # 8
        h0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        c0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
            torch.device("cpu"))
        outs = None
        for i in range(seq_len):
            out = self.encode_region(known_frame[i])
            out = out.unsqueeze(1)
            if outs is None:
                outs = out
            else:
                outs = torch.cat((outs, out), dim=1)
        lstm_neighbor_output, (hn, cn) = self.lstm_neighbor(outs, (h0, c0))
        hn_neighbor = hn.squeeze(0)
        hn_neighbor = F.relu(self.lstm_neighbor_fc(hn_neighbor))
        return hn_neighbor

    def encode_region(self, region_frame):
        # out_frame = F.relu(self.frame_nb_con1(region_frame))
        # out_frame = F.relu(self.frame_nb_con2(out_frame))

        # out_frame = region_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_nb_fc1(region_frame))
        out_frame = F.relu(self.frame_nb_fc2(out_frame))

        return out_frame

    def encode_observation_map(self, frame):
        out_frame = F.relu(self.frame_con1(frame))
        # out_frame = F.relu(self.frame_con2(out_frame))
        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        return out_frame

    def forward(self, state):
        frame, robot_pose = state
        batch_size = robot_pose.shape[0]
        seq_len = robot_pose.shape[1]
        robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
        out_frame = self.encode_observation_map(frame)
        frame_spacial = self.encode_neighbor(frame)
        outs = torch.cat((out_frame, frame_spacial), dim=1)

        out = F.relu(self.pose_fc3(outs))

        val = self.fc_val(out)
        return val


# class DQN_Network11_Temporal_Spacial1_2(torch.nn.Module):
#     """spacial lstm不管深度，只管方向"""
#
#     def __init__(self, action_size):
#         super().__init__()
#         # 3 5 4 2
#         # 15 36 18
#         self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
#         self.frame_fc1 = torch.nn.Linear(3888, 784)
#         self.frame_fc2 = torch.nn.Linear(784, 128)
#
#         self.frame_nb_fc1 = torch.nn.Linear(972, 384)
#         self.frame_nb_fc2 = torch.nn.Linear(384, 64)
#
#         self.hn_neighbor_state_dim = 384
#         self.lstm_neighbor = torch.nn.LSTM(64, self.hn_neighbor_state_dim, batch_first=True)
#         self.lstm_neighbor_fc = torch.nn.Linear(self.hn_neighbor_state_dim, 128)
#         # lstm neighbor output = 128
#
#         self.pose_fc1a = torch.nn.Linear(3, 32)
#         self.pose_fc2a = torch.nn.Linear(32, 64)
#
#         self.pose_fc1b = torch.nn.Linear(3, 32)
#         self.pose_fc2b = torch.nn.Linear(32, 64)
#
#         self.hn_temporal_state_dim = 256
#         self.lstm_temporal = torch.nn.LSTM(128, self.hn_temporal_state_dim, batch_first=True)
#         self.lstm_temporal_fc = torch.nn.Linear(self.hn_temporal_state_dim, 128)
#
#         self.pose_fc3 = torch.nn.Linear(128 * 3, 128)
#
#         self.fc_val = torch.nn.Linear(128, action_size)
#
#     def init_weights(self):
#         for m in self.modules():
#             if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
#                 torch.nn.init.zeros_(m.weight)
#                 torch.nn.init.zeros_(m.bias)
#
#     def encode_neighbor(self, frame):
#         # 15 36 18
#         batch_size, depth, w, h = frame.shape
#         frame = torch.reshape(frame, (batch_size, 3, 5, 2, 18, 18))
#         # bs, 3, 5, 4, 2, 15, 15
#         frame = torch.transpose(frame, 1, 2)  # bs, 5, 3, 2
#         frame = torch.transpose(frame, 2, 3)  # bs, 5, 2, 3
#         frame = torch.reshape(frame, (batch_size, 10, 3, 18, 18))
#         frame = torch.reshape(frame, (batch_size, 10, -1))
#
#         # bs, 40, 3, 15, 15
#         frame = torch.transpose(frame, 0, 1)  #
#         # 40, bs, 3, 15, 15
#         # 5 4 2
#         seq_len = frame.shape[0]
#         h0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
#             torch.device("cpu"))
#         c0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
#             torch.device("cpu"))
#         outs = None
#         for i in range(seq_len):
#             out = self.encode_region(frame[i])
#             out = out.unsqueeze(1)
#             if outs is None:
#                 outs = out
#             else:
#                 outs = torch.cat((outs, out), dim=1)
#         lstm_neighbor_output, (hn, cn) = self.lstm_neighbor(outs, (h0, c0))
#         hn_neighbor = hn.squeeze(0)
#         hn_neighbor = F.relu(self.lstm_neighbor_fc(hn_neighbor))
#         return hn_neighbor
#
#     def encode_region(self, region_frame):
#         # out_frame = F.relu(self.frame_nb_con1(region_frame))
#         # out_frame = F.relu(self.frame_nb_con2(out_frame))
#
#         # out_frame = region_frame.reshape(out_frame.size()[0], -1)
#         out_frame = F.relu(self.frame_nb_fc1(region_frame))
#         out_frame = F.relu(self.frame_nb_fc2(out_frame))
#
#         return out_frame
#
#     def encode_pos(self, robot_pose):
#         out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
#         out_pose_a = F.relu(self.pose_fc2a(out_pose_a))
#
#         out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
#         out_pose_b = F.relu(self.pose_fc2b(out_pose_b))
#
#         out = torch.cat((out_pose_a, out_pose_b), dim=1)
#         return out
#
#     def encode_observation_map(self, frame):
#         out_frame = F.relu(self.frame_con1(frame))
#         # out_frame = F.relu(self.frame_con2(out_frame))
#         out_frame = out_frame.reshape(out_frame.size()[0], -1)
#         out_frame = F.relu(self.frame_fc1(out_frame))
#         out_frame = F.relu(self.frame_fc2(out_frame))
#
#         return out_frame
#
#     def encode_temporal_relation(self, robot_pose_reshape, batch_size, seq_len):
#         outs = None
#         h0 = torch.zeros(1, batch_size, self.hn_temporal_state_dim).to(
#             torch.device("cpu"))
#         c0 = torch.zeros(1, batch_size, self.hn_temporal_state_dim).to(
#             torch.device("cpu"))
#         for i in range(seq_len):
#             out = self.encode_pos(robot_pose_reshape[i])
#             out = out.unsqueeze(1)
#             if outs is None:
#                 outs = out
#             else:
#                 outs = torch.cat((outs, out), dim=1)
#         lstm_temporal_output, (hn, cn) = self.lstm_temporal(outs, (h0, c0))
#         hn = hn.squeeze(0)
#         hn = F.relu(self.lstm_temporal_fc(hn))
#         return hn
#
#     def forward(self, state):
#         frame, robot_pose = state
#         batch_size = robot_pose.shape[0]
#         seq_len = robot_pose.shape[1]
#         robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
#         out_frame = self.encode_observation_map(frame)
#         pose_temporal = self.encode_temporal_relation(robot_pose_reshape, batch_size, seq_len)
#         frame_spacial = self.encode_neighbor(frame)
#         outs = torch.cat((pose_temporal, out_frame, frame_spacial), dim=1)
#
#         out = F.relu(self.pose_fc3(outs))
#
#         val = self.fc_val(out)
#         return val
#
#
# class DQN_Network11_Temporal_Spacial2(torch.nn.Module):
#     def __init__(self, action_size):
#         super().__init__()
#         # 3 5 4 2
#         # 15 36 18
#         self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
#         self.frame_fc1 = torch.nn.Linear(3888, 784)
#         self.frame_fc2 = torch.nn.Linear(784, 128)
#
#         self.frame_nb_fc1 = torch.nn.Linear(1215, 384)
#         self.frame_nb_fc2 = torch.nn.Linear(384, 64)
#
#         self.hn_neighbor_state_dim = 384
#         self.lstm_neighbor = torch.nn.LSTM(65, self.hn_neighbor_state_dim, batch_first=True)
#         self.lstm_neighbor_fc = torch.nn.Linear(self.hn_neighbor_state_dim, 128)
#         # lstm neighbor output = 128
#
#         self.pose_fc1a = torch.nn.Linear(3, 32)
#         self.pose_fc2a = torch.nn.Linear(32, 64)
#
#         self.pose_fc1b = torch.nn.Linear(3, 32)
#         self.pose_fc2b = torch.nn.Linear(32, 64)
#
#         self.hn_temporal_state_dim = 256
#         self.lstm_temporal = torch.nn.LSTM(128, self.hn_temporal_state_dim, batch_first=True)
#         self.lstm_temporal_fc = torch.nn.Linear(self.hn_temporal_state_dim, 128)
#
#         self.pose_fc3 = torch.nn.Linear(128 * 3, 128)
#
#         self.fc_val = torch.nn.Linear(128, action_size)
#
#     def init_weights(self):
#         for m in self.modules():
#             if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
#                 torch.nn.init.zeros_(m.weight)
#                 torch.nn.init.zeros_(m.bias)
#
#     def encode_neighbor(self, frame):
#         # 15 36 18
#         batch_size, depth, w, h = frame.shape
#         frame = torch.reshape(frame, (batch_size, 15, 4, 9, 2, 9))
#         # bs, 3, 5, 4, 2, 15, 15
#         frame = torch.transpose(frame, 1, 2)  # batch_size, 4, 15, 9, 2, 9
#         frame = torch.transpose(frame, 3, 4)  # batch_size, 4, 15, 2, 9, 9
#         frame = torch.transpose(frame, 2, 3)  # batch_size, 4, 2, 15, 9, 9
#         frame = torch.reshape(frame, shape=(-1, 8, 3, 405))
#         frame = torch.transpose(frame, 0, 1)  # 8, bs, 15, 9, 9
#         # 在8这个维度上，对batch中的每一个，求出target最多的
#         known_frame = frame[:, :, 2, :]  # 8, bs, 405
#         regions_target_num = torch.sum(known_frame, dim=-1).unsqueeze(-1)  # 8, bs
#         # 从大到小排列，free cells少的最好，排在后面
#         index = torch.argsort(regions_target_num, dim=0, descending=False)  # 8, bs, 1
#         index = index.repeat(1, 1, 1215)
#         frame = torch.reshape(frame, shape=(8, -1, 1215))  # 8, bs, 1215
#         sorted_frame = torch.gather(frame, dim=0, index=index)
#         seq_len = frame.shape[0]
#         h0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
#             torch.device("cpu"))
#         c0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
#             torch.device("cpu"))
#         outs = None
#         # 10*972
#         for i in range(seq_len):
#             out = self.encode_region(sorted_frame[i])
#             ind = index[i, :, 0].unsqueeze(1)
#             out = torch.cat((out, ind), dim=1)
#             out = out.unsqueeze(1)
#             if outs is None:
#                 outs = out
#             else:
#                 outs = torch.cat((outs, out), dim=1)
#         lstm_neighbor_output, (hn, cn) = self.lstm_neighbor(outs, (h0, c0))
#         hn_neighbor = hn.squeeze(0)
#         hn_neighbor = F.relu(self.lstm_neighbor_fc(hn_neighbor))
#         return hn_neighbor
#
#     def encode_region(self, region_frame):
#         # out_frame = F.relu(self.frame_nb_con1(region_frame))
#         # out_frame = F.relu(self.frame_nb_con2(out_frame))
#
#         # out_frame = region_frame.reshape(out_frame.size()[0], -1)
#         out_frame = F.relu(self.frame_nb_fc1(region_frame))
#         out_frame = F.relu(self.frame_nb_fc2(out_frame))
#
#         return out_frame
#
#     def encode_pos(self, robot_pose):
#         out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
#         out_pose_a = F.relu(self.pose_fc2a(out_pose_a))
#
#         out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
#         out_pose_b = F.relu(self.pose_fc2b(out_pose_b))
#
#         out = torch.cat((out_pose_a, out_pose_b), dim=1)
#         return out
#
#     def encode_observation_map(self, frame):
#         out_frame = F.relu(self.frame_con1(frame))
#         # out_frame = F.relu(self.frame_con2(out_frame))
#         out_frame = out_frame.reshape(out_frame.size()[0], -1)
#         out_frame = F.relu(self.frame_fc1(out_frame))
#         out_frame = F.relu(self.frame_fc2(out_frame))
#
#         return out_frame
#
#     def encode_temporal_relation(self, robot_pose_reshape, batch_size, seq_len):
#         outs = None
#         h0 = torch.zeros(1, batch_size, self.hn_temporal_state_dim).to(
#             torch.device("cpu"))
#         c0 = torch.zeros(1, batch_size, self.hn_temporal_state_dim).to(
#             torch.device("cpu"))
#         for i in range(seq_len):
#             out = self.encode_pos(robot_pose_reshape[i])
#             out = out.unsqueeze(1)
#             if outs is None:
#                 outs = out
#             else:
#                 outs = torch.cat((outs, out), dim=1)
#         lstm_temporal_output, (hn, cn) = self.lstm_temporal(outs, (h0, c0))
#         hn = hn.squeeze(0)
#         hn = F.relu(self.lstm_temporal_fc(hn))
#         return hn
#
#     def forward(self, state):
#         frame, robot_pose = state
#         batch_size = robot_pose.shape[0]
#         seq_len = robot_pose.shape[1]
#         robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
#         out_frame = self.encode_observation_map(frame)
#         pose_temporal = self.encode_temporal_relation(robot_pose_reshape, batch_size, seq_len)
#         frame_spacial = self.encode_neighbor(frame)
#         outs = torch.cat((pose_temporal, out_frame, frame_spacial), dim=1)
#
#         out = F.relu(self.pose_fc3(outs))
#
#         val = self.fc_val(out)
#         return val
