import torch
from torch import nn
import torch.nn.functional as F


class DQN_Network11_KnownMap(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
        self.frame_fc1 = torch.nn.Linear(3888, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.knownmap_con1 = torch.nn.Conv2d(20, 32, kernel_size=4, stride=2, padding=1)

        self.knownmap_fc1 = torch.nn.Linear(5184, 1024)
        self.knownmap_fc2 = torch.nn.Linear(1024, 512)
        self.knownmap_fc3 = torch.nn.Linear(512, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(128 * 2 + 64 * 2, 128)
        # self.pose_fc3 = torch.nn.Linear(128 + 64 * 2, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        global_map, frame, robot_pose = state

        out_frame = F.relu(self.frame_con1(frame))
        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        out_map = F.relu(self.knownmap_con1(global_map))
        out_map = out_map.reshape(out_map.size()[0], -1)
        out_map = F.relu(self.knownmap_fc1(out_map))
        out_map = F.relu(self.knownmap_fc2(out_map))
        out_map = F.relu(self.knownmap_fc3(out_map))

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        # out = torch.cat((out_map, out_frame, out_pose_a, out_pose_b), dim=1)
        out = torch.cat((out_map, out_frame, out_pose_a, out_pose_b), dim=1)

        # print(out.shape)
        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.fc_val(out)
        return val


class DQN_Network11_KnownMap_Temporal(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(15, 24, kernel_size=4, stride=2, padding=1)
        self.frame_fc1 = torch.nn.Linear(3888, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.knownmap_con1 = torch.nn.Conv2d(250, 50, kernel_size=4, stride=2, padding=1)

        self.knownmap_fc1 = torch.nn.Linear(8100, 4096)
        self.knownmap_fc2 = torch.nn.Linear(4096, 1024)
        self.knownmap_fc3 = torch.nn.Linear(1024, 128)

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(128 * 2 + 64 * 2, 128)
        # self.pose_fc3 = torch.nn.Linear(128 + 64 * 2, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        global_map, frame, robot_pose = state

        out_frame = F.relu(self.frame_con1(frame))
        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        out_map = F.relu(self.knownmap_con1(global_map))
        out_map = out_map.reshape(out_map.size()[0], -1)
        out_map = F.relu(self.knownmap_fc1(out_map))
        out_map = F.relu(self.knownmap_fc2(out_map))
        out_map = F.relu(self.knownmap_fc3(out_map))

        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        # out = torch.cat((out_map, out_frame, out_pose_a, out_pose_b), dim=1)
        out = torch.cat((out_map, out_frame, out_pose_a, out_pose_b), dim=1)

        # print(out.shape)
        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.fc_val(out)
        return val