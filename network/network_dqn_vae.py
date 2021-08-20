import pfrl
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from utilities.util import compute_conv_out_width, compute_conv_out_node_num


class VAE_CNN_P3D(nn.Module):
    def __init__(self):
        super(VAE_CNN_P3D, self).__init__()
        in_channels = 15
        out_channels = 24
        kernel_size = 4
        stride = 2
        padding = 1
        w = 36
        h = 18
        out_w = compute_conv_out_width(i=w, k=kernel_size, s=stride, p=padding)
        out_h = compute_conv_out_width(i=h, k=kernel_size, s=stride, p=padding)
        out_node_num = compute_conv_out_node_num(d=out_channels, w=out_w, h=out_h)

        self.encoder_conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),  # (b, 24, 9, 18)
            nn.ReLU(True),
        )
        self.encoder_linear_layer = nn.Sequential(
            nn.Linear(out_node_num, 512),
            nn.ReLU(True),
        )
        self.fc_mu = torch.nn.Linear(512, 128)
        self.fc_std = torch.nn.Linear(512, 128)

        self.decoder_linear_layer = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, out_node_num),
            nn.ReLU(True)
        )
        # 24 * 9 * 18
        self.decoder_conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size,
                               stride=stride,
                               padding=padding),  # (b, 8, 16, 16)
            nn.Tanh()
        )

    def encoder(self, x):
        h1_conv = self.encoder_conv_layer(x)
        self.batch_size = h1_conv.size()[0]
        h1_linear = h1_conv.view(self.batch_size, -1)
        h2 = self.encoder_linear_layer(h1_linear)
        mu = self.fc_mu(h2)
        logvar = self.fc_std(h2)
        return mu, logvar

    def decoder(self, z):
        h3_linear = self.decoder_linear_layer(z)
        h3_conv = h3_linear.view(self.batch_size, 24, 8, 8)
        x = self.decoder_conv_layer(h3_conv)
        return x

    # 重新参数化
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # 计算标准差
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()  # 从标准的正态分布中随机采样一个eps
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return mu, logvar, self.decoder(z)


class DQN_Network11_VAE(torch.nn.Module):
    def __init__(self, action_size, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.in_channels = 15
        self.out_channels = 24
        kernel_size = 4
        stride = 2
        padding = 1
        self.w = 36
        self.h = 18
        self.out_w = compute_conv_out_width(i=self.w, k=kernel_size, s=stride, p=padding)
        self.out_h = compute_conv_out_width(i=self.h, k=kernel_size, s=stride, p=padding)
        out_node_num = compute_conv_out_node_num(d=self.out_channels, w=self.out_w, h=self.out_h)
        # encoder
        self.encoder_conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),  # (b, 24, 9, 18)
            nn.ReLU(True),
        )
        self.encoder_linear_layer = nn.Sequential(
            nn.Linear(out_node_num, 1024),
            nn.ReLU(True),
        )

        self.fc_mu = torch.nn.Linear(1024, 512)
        self.fc_std = torch.nn.Linear(1024, 512)
        # decoder
        self.decoder_linear_layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, out_node_num),
            nn.ReLU(True)
        )
        # 24 * 9 * 18
        self.decoder_conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.in_channels * 12,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding),  # (b, 8, 16, 16)
            nn.Tanh()
        )

        # dqn
        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(640, 256)

        self.pose_fc4 = torch.nn.Linear(256, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        frame, robot_pose = state

        # encode
        mu, logvar = self.encoder(frame)
        z = self.reparametrize(mu, logvar)

        out_decoder = self.decoder(z)

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out = torch.cat((mu, out_pose_a, out_pose_b), dim=1)
        # print(out.shape)
        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.fc_val(out)
        return val, out_decoder, mu, logvar

    def encoder(self, x):
        h1_conv = self.encoder_conv_layer(x)
        self.batch_size = h1_conv.size()[0]
        h1_linear = h1_conv.view(self.batch_size, -1)
        h2 = self.encoder_linear_layer(h1_linear)
        mu = self.fc_mu(h2)
        logvar = self.fc_std(h2)
        return mu, logvar

    def decoder(self, z):
        h3_linear = self.decoder_linear_layer(z)
        h3_conv = h3_linear.view(self.batch_size, self.out_channels, self.out_w, self.out_h)
        x = self.decoder_conv_layer(h3_conv)
        x = x.view(self.batch_size, 12, self.in_channels, self.w, self.h)
        return x

    # 重新参数化
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # 计算标准差
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()  # 从标准的正态分布中随机采样一个eps
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


class DQN_Network11_VAE_Time_enhanced3(torch.nn.Module):
    def __init__(self, action_size, batch_size=128):
        super().__init__()
        self.in_channels = 15
        self.out_channels = 24
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.w = 36
        self.h = 18
        self.out_w = compute_conv_out_width(i=self.w, k=self.kernel_size, s=self.stride, p=self.padding)
        self.out_h = compute_conv_out_width(i=self.h, k=self.kernel_size, s=self.stride, p=self.padding)
        self.out_node_num = compute_conv_out_node_num(d=self.out_channels, w=self.out_w, h=self.out_h)

        # 编码
        self.init_encode_layer()

        self.fc_mu = torch.nn.Linear(1024, 512)
        self.fc_std = torch.nn.Linear(1024, 512)

        # 解码层
        self.init_decoder_layer()

        # dqn
        # self.pose_fc3 = torch.nn.Linear(256 * 5, 256 * 3)

        self.pose_fc4 = torch.nn.Linear(1024 + 128 * 2, 448)

        self.pose_fc5 = torch.nn.Linear(448, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_encode_layer(self):
        self.frame_con1 = torch.nn.Conv2d(15, 24, self.kernel_size, self.stride, self.padding)
        self.frame_fc1 = torch.nn.Linear(3888, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.hn_neighbor_state_dim = 512
        self.lstm_neighbor1 = nn.LSTM(128, self.hn_neighbor_state_dim, batch_first=True)
        self.lstm_neighbor2 = nn.LSTM(128, self.hn_neighbor_state_dim, batch_first=True)

    def init_decoder_layer(self):
        self.decoder_linear_layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.out_node_num),
            nn.ReLU(True)
        )
        # 24 * 9 * 18
        self.decoder_conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.in_channels * 12,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding),  # (b, 8, 16, 16)
            nn.Tanh()
        )

    def encode(self, frame_reshape, robot_pose_reshape):
        frame_outs = None
        robot_pose_outs = None
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

            out_pose_a = F.relu(self.pose_fc1a(robot_pose_reshape[i][:, 0:3]))
            out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

            out_pose_b = F.relu(self.pose_fc1b(robot_pose_reshape[i][:, 3:6]))
            out_pose_b = F.relu(self.pose_fc2b(out_pose_b))
            out_frame = out_frame.unsqueeze(1)
            robot_pose_cat_out = torch.cat((out_pose_a, out_pose_b), dim=1).unsqueeze(1)

            if frame_outs is None:
                # batch_size * hidden_size
                frame_outs = out_frame
                robot_pose_outs = robot_pose_cat_out
            else:
                frame_outs = torch.cat((frame_outs, out_frame), dim=1)
                robot_pose_outs = torch.cat((robot_pose_outs, robot_pose_cat_out), dim=1)
        lstm_neighbor_output, (hn_frame, cn) = self.lstm_neighbor1(frame_outs, (h0, c0))
        lstm_neighbor_output, (hn_robot_pose, cn) = self.lstm_neighbor2(robot_pose_outs, (h0, c0))

        hn_frame = hn_frame.squeeze(0)
        hn_robot_pose = hn_robot_pose.squeeze(0)
        hn = torch.cat((hn_frame, hn_robot_pose), dim=1)
        # 输出是batch_size * 1024
        return hn

    def encode_frame(self, frame_reshape):
        out_frame = F.relu(self.frame_con1(frame_reshape[-1]))
        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))
        return out_frame

    def encode_robot_pose(self, robot_pose_reshape):
        out_pose_a = F.relu(self.pose_fc1a(robot_pose_reshape[-1][:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose_reshape[-1][:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))
        out_pose = torch.cat((out_pose_a, out_pose_b), dim=1)
        return out_pose

    def decoder(self, z):
        h3_linear = self.decoder_linear_layer(z)
        h3_conv = h3_linear.view(z.shape[0], self.out_channels, self.out_w, self.out_h)
        x = self.decoder_conv_layer(h3_conv)
        x = x.view(z.shape[0], 12, self.in_channels, self.w, self.h)
        return x

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # 计算标准差
        # if torch.cuda.is_available():
        #     eps = torch.cuda.FloatTensor(std.size()).normal_()  # 从标准的正态分布中随机采样一个eps
        # else:
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, state):
        frame, robot_pose = state
        frame_reshape = torch.transpose(frame, 0, 1)
        robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
        # 经过了lstm
        hn = self.encode(frame_reshape, robot_pose_reshape)

        mu = self.fc_mu(hn)
        logvar = self.fc_std(hn)

        # 一个分支是解码
        z = self.reparametrize(mu, logvar)
        out_decoder = self.decoder(z)

        # 一个是DQN
        h_frame = self.encode_frame(frame_reshape)
        h_pose = self.encode_robot_pose(robot_pose_reshape)
        out = torch.cat((hn, h_frame, h_pose), dim=1)

        out = F.relu(self.pose_fc4(out))
        out = F.relu(self.pose_fc5(out))

        val = self.fc_val(out)
        return val, out_decoder, mu, logvar


class DQN_Network11_Time_enhanced3(torch.nn.Module):
    def __init__(self, action_size, batch_size=128):
        super().__init__()
        self.in_channels = 15
        self.out_channels = 24
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.w = 36
        self.h = 18
        self.out_w = compute_conv_out_width(i=self.w, k=self.kernel_size, s=self.stride, p=self.padding)
        self.out_h = compute_conv_out_width(i=self.h, k=self.kernel_size, s=self.stride, p=self.padding)
        self.out_node_num = compute_conv_out_node_num(d=self.out_channels, w=self.out_w, h=self.out_h)

        # 编码
        self.init_encode_layer()

        # self.fc_mu = torch.nn.Linear(1024, 512)
        # self.fc_std = torch.nn.Linear(1024, 512)

        # dqn
        # self.pose_fc3 = torch.nn.Linear(256 * 5, 256 * 3)

        self.pose_fc4 = torch.nn.Linear(1024 + 128 * 2, 448)

        self.pose_fc5 = torch.nn.Linear(448, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_encode_layer(self):
        self.frame_con1 = torch.nn.Conv2d(15, 24, self.kernel_size, self.stride, self.padding)
        self.frame_fc1 = torch.nn.Linear(3888, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.hn_neighbor_state_dim = 512
        self.lstm_neighbor1 = nn.LSTM(128, self.hn_neighbor_state_dim, batch_first=True)
        self.lstm_neighbor2 = nn.LSTM(128, self.hn_neighbor_state_dim, batch_first=True)


    def encode(self, frame_reshape, robot_pose_reshape):
        frame_outs = None
        robot_pose_outs = None
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

            out_pose_a = F.relu(self.pose_fc1a(robot_pose_reshape[i][:, 0:3]))
            out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

            out_pose_b = F.relu(self.pose_fc1b(robot_pose_reshape[i][:, 3:6]))
            out_pose_b = F.relu(self.pose_fc2b(out_pose_b))
            out_frame = out_frame.unsqueeze(1)
            robot_pose_cat_out = torch.cat((out_pose_a, out_pose_b), dim=1).unsqueeze(1)

            if frame_outs is None:
                # batch_size * hidden_size
                frame_outs = out_frame
                robot_pose_outs = robot_pose_cat_out
            else:
                frame_outs = torch.cat((frame_outs, out_frame), dim=1)
                robot_pose_outs = torch.cat((robot_pose_outs, robot_pose_cat_out), dim=1)
        lstm_neighbor_output, (hn_frame, cn) = self.lstm_neighbor1(frame_outs, (h0, c0))
        lstm_neighbor_output, (hn_robot_pose, cn) = self.lstm_neighbor2(robot_pose_outs, (h0, c0))

        hn_frame = hn_frame.squeeze(0)
        hn_robot_pose = hn_robot_pose.squeeze(0)
        hn = torch.cat((hn_frame, hn_robot_pose), dim=1)
        # 输出是batch_size * 1024
        return hn

    def encode_frame(self, frame_reshape):
        out_frame = F.relu(self.frame_con1(frame_reshape[-1]))
        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))
        return out_frame

    def encode_robot_pose(self, robot_pose_reshape):
        out_pose_a = F.relu(self.pose_fc1a(robot_pose_reshape[-1][:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose_reshape[-1][:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))
        out_pose = torch.cat((out_pose_a, out_pose_b), dim=1)
        return out_pose

    def forward(self, state):
        frame, robot_pose = state
        frame_reshape = torch.transpose(frame, 0, 1)
        robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
        # 经过了lstm
        hn = self.encode(frame_reshape, robot_pose_reshape)

        # 一个是DQN
        h_frame = self.encode_frame(frame_reshape)
        h_pose = self.encode_robot_pose(robot_pose_reshape)
        out = torch.cat((hn, h_frame, h_pose), dim=1)

        out = F.relu(self.pose_fc4(out))
        out = F.relu(self.pose_fc5(out))

        val = self.fc_val(out)
        return val


# class DQN_Network11_VAE_Time_enhanced2(torch.nn.Module):
#     def __init__(self, action_size, batch_size=128):
#         super().__init__()
#         self.in_channels = 15
#         self.out_channels = 24
#         self.kernel_size = 4
#         self.stride = 2
#         self.padding = 1
#         self.w = 36
#         self.h = 18
#         self.out_w = compute_conv_out_width(i=self.w, k=self.kernel_size, s=self.stride, p=self.padding)
#         self.out_h = compute_conv_out_width(i=self.h, k=self.kernel_size, s=self.stride, p=self.padding)
#         self.out_node_num = compute_conv_out_node_num(d=self.out_channels, w=self.out_w, h=self.out_h)
#
#         # 编码
#         self.init_encode_layer()
#
#         self.fc_mu = torch.nn.Linear(1024, 512)
#         self.fc_std = torch.nn.Linear(1024, 512)
#
#         # 解码层
#         self.init_decoder_layer()
#
#         # dqn
#         # self.pose_fc3 = torch.nn.Linear(256 * 5, 256 * 3)
#
#         self.pose_fc4 = torch.nn.Linear(1024 + 128, 448)
#
#         self.pose_fc5 = torch.nn.Linear(448, 32)
#
#         self.fc_val = torch.nn.Linear(32, action_size)
#
#     def init_encode_layer(self):
#         self.frame_con1 = torch.nn.Conv2d(15, 24, self.kernel_size, self.stride, self.padding)
#         self.frame_fc1 = torch.nn.Linear(3888, 512)
#         self.frame_fc2 = torch.nn.Linear(512, 128)
#
#         self.pose_fc1a = torch.nn.Linear(3, 32)
#         self.pose_fc2a = torch.nn.Linear(32, 64)
#
#         self.pose_fc1b = torch.nn.Linear(3, 32)
#         self.pose_fc2b = torch.nn.Linear(32, 64)
#
#         self.hn_neighbor_state_dim = 512
#         self.lstm_neighbor1 = nn.LSTM(128, self.hn_neighbor_state_dim, batch_first=True)
#         self.lstm_neighbor2 = nn.LSTM(128, self.hn_neighbor_state_dim, batch_first=True)
#
#     def init_decoder_layer(self):
#         self.decoder_linear_layer = nn.Sequential(
#             nn.Linear(512, 1024),
#             nn.ReLU(True),
#             nn.Linear(1024, self.out_node_num),
#             nn.ReLU(True)
#         )
#         # 24 * 9 * 18
#         self.decoder_conv_layer = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.in_channels * 12,
#                                kernel_size=self.kernel_size,
#                                stride=self.stride,
#                                padding=self.padding),  # (b, 8, 16, 16)
#             nn.Tanh()
#         )
#
#     def encode(self, frame_reshape, robot_pose_reshape):
#         frame_outs = None
#         robot_pose_outs = None
#         batch_size = frame_reshape.shape[1]
#         h0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
#             torch.device("cpu"))
#         c0 = torch.zeros(1, batch_size, self.hn_neighbor_state_dim).to(
#             torch.device("cpu"))
#
#         for i in range(5):
#             out_frame = F.relu(self.frame_con1(frame_reshape[i]))
#             out_frame = out_frame.reshape(out_frame.size()[0], -1)
#             out_frame = F.relu(self.frame_fc1(out_frame))
#             out_frame = F.relu(self.frame_fc2(out_frame))
#
#             out_pose_a = F.relu(self.pose_fc1a(robot_pose_reshape[i][:, 0:3]))
#             out_pose_a = F.relu(self.pose_fc2a(out_pose_a))
#
#             out_pose_b = F.relu(self.pose_fc1b(robot_pose_reshape[i][:, 3:6]))
#             out_pose_b = F.relu(self.pose_fc2b(out_pose_b))
#             out_frame = out_frame.unsqueeze(1)
#             robot_pose_cat_out = torch.cat((out_pose_a, out_pose_b), dim=1).unsqueeze(1)
#
#             if frame_outs is None:
#                 # batch_size * hidden_size
#                 frame_outs = out_frame
#                 robot_pose_outs = robot_pose_cat_out
#             else:
#                 frame_outs = torch.cat((frame_outs, out_frame), dim=1)
#                 robot_pose_outs = torch.cat((robot_pose_outs, robot_pose_cat_out), dim=1)
#         lstm_neighbor_output, (hn_frame, cn) = self.lstm_neighbor1(frame_outs, (h0, c0))
#         lstm_neighbor_output, (hn_robot_pose, cn) = self.lstm_neighbor2(robot_pose_outs, (h0, c0))
#
#         hn_frame = hn_frame.squeeze(0)
#         hn_robot_pose = hn_robot_pose.squeeze(0)
#         hn = torch.cat((hn_frame, hn_robot_pose), dim=1)
#         # 输出是batch_size * 1024
#         return hn
#
#     def encode_frame(self, frame_reshape):
#         out_frame = F.relu(self.frame_con1(frame_reshape[-1]))
#         out_frame = out_frame.reshape(out_frame.size()[0], -1)
#         out_frame = F.relu(self.frame_fc1(out_frame))
#         out_frame = F.relu(self.frame_fc2(out_frame))
#         return out_frame
#
#     def decoder(self, z):
#         h3_linear = self.decoder_linear_layer(z)
#         h3_conv = h3_linear.view(z.shape[0], self.out_channels, self.out_w, self.out_h)
#         x = self.decoder_conv_layer(h3_conv)
#         x = x.view(z.shape[0], 12, self.in_channels, self.w, self.h)
#         return x
#
#     def reparametrize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()  # 计算标准差
#         # if torch.cuda.is_available():
#         #     eps = torch.cuda.FloatTensor(std.size()).normal_()  # 从标准的正态分布中随机采样一个eps
#         # else:
#         eps = torch.FloatTensor(std.size()).normal_()
#         eps = Variable(eps)
#         return eps.mul(std).add_(mu)
#
#     def forward(self, state):
#         frame, robot_pose = state
#         frame_reshape = torch.transpose(frame, 0, 1)
#         robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
#         # 经过了lstm
#         hn = self.encode(frame_reshape, robot_pose_reshape)
#
#         mu = self.fc_mu(hn)
#         logvar = self.fc_std(hn)
#
#         # 一个分支是解码
#         z = self.reparametrize(mu, logvar)
#         out_decoder = self.decoder(z)
#
#         # 一个是DQN
#         h_frame = self.encode_frame(frame_reshape)
#         out = torch.cat((hn, h_frame), dim=1)
#
#         out = F.relu(self.pose_fc4(out))
#         out = F.relu(self.pose_fc5(out))
#
#         val = self.fc_val(out)
#         return val, out_decoder, mu, logvar

class DQN_Network11_VAE_Time_enhanced2(torch.nn.Module):
    def __init__(self, action_size, batch_size=128):
        super().__init__()
        self.in_channels = 15
        self.out_channels = 24
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.w = 36
        self.h = 18
        self.out_w = compute_conv_out_width(i=self.w, k=self.kernel_size, s=self.stride, p=self.padding)
        self.out_h = compute_conv_out_width(i=self.h, k=self.kernel_size, s=self.stride, p=self.padding)
        self.out_node_num = compute_conv_out_node_num(d=self.out_channels, w=self.out_w, h=self.out_h)

        # 编码
        self.init_encode_layer()

        self.fc_mu = torch.nn.Linear(1024, 512)
        self.fc_std = torch.nn.Linear(1024, 512)

        # 解码层
        self.init_decoder_layer()

        # dqn
        # self.pose_fc3 = torch.nn.Linear(256 * 5, 256 * 3)

        self.pose_fc4 = torch.nn.Linear(1024 + 128, 448)

        self.pose_fc5 = torch.nn.Linear(448, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_encode_layer(self):
        self.frame_con1 = torch.nn.Conv2d(15, 24, self.kernel_size, self.stride, self.padding)
        self.frame_fc1 = torch.nn.Linear(3888, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.hn_neighbor_state_dim = 512
        self.lstm_neighbor1 = nn.LSTM(128, self.hn_neighbor_state_dim, batch_first=True)
        self.lstm_neighbor2 = nn.LSTM(128, self.hn_neighbor_state_dim, batch_first=True)

    def init_decoder_layer(self):
        self.decoder_linear_layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.out_node_num),
            nn.ReLU(True)
        )
        # 24 * 9 * 18
        self.decoder_conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.in_channels * 12,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding),  # (b, 8, 16, 16)
            nn.Tanh()
        )

    def encode(self, frame_reshape, robot_pose_reshape):
        frame_outs = None
        robot_pose_outs = None
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

            out_pose_a = F.relu(self.pose_fc1a(robot_pose_reshape[i][:, 0:3]))
            out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

            out_pose_b = F.relu(self.pose_fc1b(robot_pose_reshape[i][:, 3:6]))
            out_pose_b = F.relu(self.pose_fc2b(out_pose_b))
            out_frame = out_frame.unsqueeze(1)
            robot_pose_cat_out = torch.cat((out_pose_a, out_pose_b), dim=1).unsqueeze(1)

            if frame_outs is None:
                # batch_size * hidden_size
                frame_outs = out_frame
                robot_pose_outs = robot_pose_cat_out
            else:
                frame_outs = torch.cat((frame_outs, out_frame), dim=1)
                robot_pose_outs = torch.cat((robot_pose_outs, robot_pose_cat_out), dim=1)
        lstm_neighbor_output, (hn_frame, cn) = self.lstm_neighbor1(frame_outs, (h0, c0))
        lstm_neighbor_output, (hn_robot_pose, cn) = self.lstm_neighbor2(robot_pose_outs, (h0, c0))

        hn_frame = hn_frame.squeeze(0)
        hn_robot_pose = hn_robot_pose.squeeze(0)
        hn = torch.cat((hn_frame, hn_robot_pose), dim=1)
        # 输出是batch_size * 1024
        return hn

    def encode_frame(self, frame_reshape):
        out_frame = F.relu(self.frame_con1(frame_reshape[-1]))
        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))
        return out_frame

    def decoder(self, z):
        h3_linear = self.decoder_linear_layer(z)
        h3_conv = h3_linear.view(z.shape[0], self.out_channels, self.out_w, self.out_h)
        x = self.decoder_conv_layer(h3_conv)
        x = x.view(z.shape[0], 12, self.in_channels, self.w, self.h)
        return x

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # 计算标准差
        # if torch.cuda.is_available():
        #     eps = torch.cuda.FloatTensor(std.size()).normal_()  # 从标准的正态分布中随机采样一个eps
        # else:
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, state):
        frame, robot_pose = state
        frame_reshape = torch.transpose(frame, 0, 1)
        robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
        # 经过了lstm
        hn = self.encode(frame_reshape, robot_pose_reshape)

        mu = self.fc_mu(hn)
        logvar = self.fc_std(hn)

        # 一个分支是解码
        z = self.reparametrize(mu, logvar)
        out_decoder = self.decoder(z)

        # 一个是DQN
        h_frame = self.encode_frame(frame_reshape)
        out = torch.cat((hn, h_frame), dim=1)

        out = F.relu(self.pose_fc4(out))
        out = F.relu(self.pose_fc5(out))

        val = self.fc_val(out)
        return val, out_decoder, mu, logvar


class DQN_Network11_VAE_Time_enhanced(torch.nn.Module):
    def __init__(self, action_size, batch_size=128):
        super().__init__()
        self.in_channels = 15
        self.out_channels = 24
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.w = 36
        self.h = 18
        self.out_w = compute_conv_out_width(i=self.w, k=self.kernel_size, s=self.stride, p=self.padding)
        self.out_h = compute_conv_out_width(i=self.h, k=self.kernel_size, s=self.stride, p=self.padding)
        self.out_node_num = compute_conv_out_node_num(d=self.out_channels, w=self.out_w, h=self.out_h)

        # 编码
        self.init_encode_layer()

        self.fc_mu = torch.nn.Linear(1024, 512)
        self.fc_std = torch.nn.Linear(1024, 512)

        # 解码层
        self.init_decoder_layer()

        # dqn
        # self.pose_fc3 = torch.nn.Linear(256 * 5, 256 * 3)

        self.pose_fc4 = torch.nn.Linear(1024 + 128, 448)

        self.pose_fc5 = torch.nn.Linear(448, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_encode_layer(self):
        self.frame_con1 = torch.nn.Conv2d(15, 24, self.kernel_size, self.stride, self.padding)
        self.frame_fc1 = torch.nn.Linear(3888, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.hn_neighbor_state_dim = 1024
        self.lstm_neighbor = nn.LSTM(256, self.hn_neighbor_state_dim, batch_first=True)

    def init_decoder_layer(self):
        self.decoder_linear_layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.out_node_num),
            nn.ReLU(True)
        )
        # 24 * 9 * 18
        self.decoder_conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.in_channels * 12,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding),  # (b, 8, 16, 16)
            nn.Tanh()
        )

    def encode(self, frame_reshape, robot_pose_reshape):
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

            out_pose_a = F.relu(self.pose_fc1a(robot_pose_reshape[i][:, 0:3]))
            out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

            out_pose_b = F.relu(self.pose_fc1b(robot_pose_reshape[i][:, 3:6]))
            out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

            cat_out = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1).unsqueeze(1)
            if outs is None:
                # batch_size * hidden_size
                outs = cat_out
            else:
                outs = torch.cat((outs, cat_out), dim=1)

        lstm_neighbor_output, (hn, cn) = self.lstm_neighbor(outs, (h0, c0))

        hn = hn.squeeze(0)
        # 输出是batch_size * 1024
        return hn

    def encode_frame(self, frame_reshape):
        out_frame = F.relu(self.frame_con1(frame_reshape[-1]))
        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))
        return out_frame

    def decoder(self, z):
        h3_linear = self.decoder_linear_layer(z)
        h3_conv = h3_linear.view(z.shape[0], self.out_channels, self.out_w, self.out_h)
        x = self.decoder_conv_layer(h3_conv)
        x = x.view(z.shape[0], 12, self.in_channels, self.w, self.h)
        return x

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # 计算标准差
        # if torch.cuda.is_available():
        #     eps = torch.cuda.FloatTensor(std.size()).normal_()  # 从标准的正态分布中随机采样一个eps
        # else:
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, state):
        frame, robot_pose = state
        frame_reshape = torch.transpose(frame, 0, 1)
        robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
        # 经过了lstm
        hn = self.encode(frame_reshape, robot_pose_reshape)

        mu = self.fc_mu(hn)
        logvar = self.fc_std(hn)

        # 一个分支是解码
        z = self.reparametrize(mu, logvar)
        out_decoder = self.decoder(z)

        # 一个是DQN
        h_frame = self.encode_frame(frame_reshape)
        out = torch.cat((hn, h_frame), dim=1)

        out = F.relu(self.pose_fc4(out))
        out = F.relu(self.pose_fc5(out))

        val = self.fc_val(out)
        return val, out_decoder, mu, logvar


class DQN_Network11_VAE_Time(torch.nn.Module):
    def __init__(self, action_size, batch_size=128):
        super().__init__()
        self.in_channels = 15
        self.out_channels = 24
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.w = 36
        self.h = 18
        self.out_w = compute_conv_out_width(i=self.w, k=self.kernel_size, s=self.stride, p=self.padding)
        self.out_h = compute_conv_out_width(i=self.h, k=self.kernel_size, s=self.stride, p=self.padding)
        self.out_node_num = compute_conv_out_node_num(d=self.out_channels, w=self.out_w, h=self.out_h)

        # 编码
        self.init_encode_layer()

        self.fc_mu = torch.nn.Linear(1024, 512)
        self.fc_std = torch.nn.Linear(1024, 512)

        # 解码层
        self.init_decoder_layer()

        # dqn
        # self.pose_fc3 = torch.nn.Linear(256 * 5, 256 * 3)

        self.pose_fc4 = torch.nn.Linear(1024, 256)

        self.pose_fc5 = torch.nn.Linear(256, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_encode_layer(self):
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
        self.frame_fc1 = torch.nn.Linear(3888, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.hn_neighbor_state_dim = 1024
        self.lstm_neighbor = nn.LSTM(256, self.hn_neighbor_state_dim, batch_first=True)

    def init_decoder_layer(self):
        self.decoder_linear_layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.out_node_num),
            nn.ReLU(True)
        )
        # 24 * 9 * 18
        self.decoder_conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.in_channels * 12,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding),  # (b, 8, 16, 16)
            nn.Tanh()
        )

    def encode(self, frame_reshape, robot_pose_reshape):
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

            out_pose_a = F.relu(self.pose_fc1a(robot_pose_reshape[i][:, 0:3]))
            out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

            out_pose_b = F.relu(self.pose_fc1b(robot_pose_reshape[i][:, 3:6]))
            out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

            cat_out = torch.cat((out_frame, out_pose_a, out_pose_b), dim=1).unsqueeze(1)
            if outs is None:
                # batch_size * hidden_size
                outs = cat_out
            else:
                outs = torch.cat((outs, cat_out), dim=1)

        lstm_neighbor_output, (hn, cn) = self.lstm_neighbor(outs, (h0, c0))
        hn = hn.squeeze(0)
        # 输出是batch_size * 1024
        return hn

    def decoder(self, z):
        h3_linear = self.decoder_linear_layer(z)
        h3_conv = h3_linear.view(z.shape[0], self.out_channels, self.out_w, self.out_h)
        x = self.decoder_conv_layer(h3_conv)
        x = x.view(z.shape[0], 12, self.in_channels, self.w, self.h)
        return x

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # 计算标准差
        # if torch.cuda.is_available():
        #     eps = torch.cuda.FloatTensor(std.size()).normal_()  # 从标准的正态分布中随机采样一个eps
        # else:
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, state):
        frame, robot_pose = state
        frame_reshape = torch.transpose(frame, 0, 1)
        robot_pose_reshape = torch.transpose(robot_pose, 0, 1)
        # 经过了lstm
        hn = self.encode(frame_reshape, robot_pose_reshape)
        mu = self.fc_mu(hn)
        logvar = self.fc_std(hn)

        # 一个分支是解码
        z = self.reparametrize(mu, logvar)
        out_decoder = self.decoder(z)

        # 一个是DQN
        out = F.relu(self.pose_fc4(hn))
        out = F.relu(self.pose_fc5(out))

        val = self.fc_val(out)
        return val, out_decoder, mu, logvar


class DQN_Network11_VAE_Time222222(torch.nn.Module):
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
