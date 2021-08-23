import torch
from torch import nn
import torch.nn.functional as F


class DQN_Network_FOV(torch.nn.Module):
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
