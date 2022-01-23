import numpy as np
import torch
import torch.nn.functional as F


class MLPDQNNetwork(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(202, 512)
        self.fc2 = torch.nn.Linear(512, 256)

        self.fc3 = torch.nn.Linear(256, 64)
        self.fc4 = torch.nn.Linear(64, action_size)

    def forward(self, state):
        [state] = state
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        val = self.fc4(out)
        return val


class MLPDQNNetwork2(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.fc_obs1 = torch.nn.Linear(200, 512)
        self.fc_obs2 = torch.nn.Linear(512, 256)
        self.fc_obs3 = torch.nn.Linear(256, 64)

        self.fc_wp1 = torch.nn.Linear(16, 64)
        self.fc_wp2 = torch.nn.Linear(64, 128)
        self.fc_wp3 = torch.nn.Linear(128, 64)

        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, action_size)

    def forward(self, state):
        [obs, wp] = state
        obs = torch.reshape(obs, (-1, obs.shape[1] * obs.shape[2]))
        obs = F.relu(self.fc_obs1(obs))
        obs = F.relu(self.fc_obs2(obs))
        obs = F.relu(self.fc_obs3(obs))

        wp = F.relu(self.fc_wp1(wp))
        wp = F.relu(self.fc_wp2(wp))
        wp = F.relu(self.fc_wp3(wp))

        out = torch.cat((obs, wp), dim=1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out


class DQNNetwork(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()

        # 100 x 100
        self.frame_con1 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.mean_pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        # 25 x 25 x 32
        self.frame_con2 = torch.nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1)
        self.max_pooling = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # 13 x 13 x 32
        self.frame_fc1 = torch.nn.Linear(1152, 512)
        self.frame_fc2 = torch.nn.Linear(512, 256)
        self.frame_fc3 = torch.nn.Linear(256, 64)

        self.pose_fc1 = torch.nn.Linear(3, 32)
        self.pose_fc2 = torch.nn.Linear(32, 128)
        self.pose_fc3 = torch.nn.Linear(128, 64)

        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        frame, wp_pose = state
        out_frame = F.relu(self.mean_pooling(self.frame_con1(frame)))
        out_frame = F.relu(self.max_pooling(self.frame_con2(out_frame)))

        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        # print("out frame shape:", out_frame.shape)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))
        out_frame = F.relu(self.frame_fc3(out_frame))

        wp_pose = F.relu(self.pose_fc1(wp_pose))
        wp_pose = F.relu(self.pose_fc2(wp_pose))
        wp_pose = F.relu(self.pose_fc3(wp_pose))

        out = torch.cat((out_frame, wp_pose), dim=1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        val = self.fc_val(out)
        return val


class DQNNetwork2(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()

        # 100 x 100
        self.frame_con1 = torch.nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.mean_pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        # 25 x 25 x 32
        self.frame_con2 = torch.nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1)
        self.max_pooling = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # 13 x 13 x 32
        self.frame_fc1 = torch.nn.Linear(1152, 256)
        self.frame_fc2 = torch.nn.Linear(256, 64)

        self.fc1 = torch.nn.Linear(64, 32)
        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        [frame, wp] = state
        out_frame = F.relu(self.mean_pooling(self.frame_con1(frame)))
        out_frame = F.relu(self.max_pooling(self.frame_con2(out_frame)))

        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        # print("out frame shape:", out_frame.shape)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        # out = torch.cat((out_frame, wp_pose), dim=1)

        out = F.relu(self.fc1(out_frame))
        val = self.fc_val(out)
        return val


class LinearDQNNetwork(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 256)
        self.fc3 = torch.nn.Linear(256, action_size)

    def forward(self, state):
        [state] = state
        out = F.relu(self.fc1(state))
        val = self.fc3(out)
        return val
