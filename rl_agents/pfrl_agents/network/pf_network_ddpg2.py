import torch
from pfrl.nn import BoundByTanh
from pfrl.policies import DeterministicHead
from torch import nn
import torch.nn.functional as F


class QNetwork2(nn.Module):
    def __init__(self, action_space):
        super(QNetwork2, self).__init__()
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

        self.fc_val = torch.nn.Linear(32, 1)

    def forward(self, state, action):
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


class PolicyNet2(nn.Module):
    def __init__(self, action_space):
        super(PolicyNet2, self).__init__()

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

        self.fc3 = torch.nn.Linear(32, len(action_space.low))
        self.bound = BoundByTanh(low=action_space.low, high=action_space.high),
        self.head = DeterministicHead()

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

        out = self.fc2(out)
        out = self.bound(out)
        out = self.head(out)
        return out
