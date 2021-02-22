import torch
import torch.nn.functional as F


class PPOPolicy3DUnknownMap2(torch.nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(3, 16, kernel_size=4, stride=2)
        self.frame_con2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(288, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1 = torch.nn.Linear(7, 32)
        self.pose_fc2 = torch.nn.Linear(32, 128)

        self.concat_fc = torch.nn.Linear(256, 64)

        self.fc_val = torch.nn.Linear(64, 1)
        self.fc_pol = torch.nn.Linear(64, action_space)

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

        val = self.fc_val(out)
        pol = self.fc_pol(out)
        pol = F.softmax(pol, dim=1)
        return val, pol


class PPOPolicy3DUnknownMap(torch.nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.fc1 = torch.nn.Linear(324, 128)
        self.fc2 = torch.nn.Linear(128, 32)

        self.fc_pose = torch.nn.Linear(7, 32)

        self.fc_val = torch.nn.Linear(64, 1)
        self.fc_pol = torch.nn.Linear(64, action_space)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, frame, robot_pose):
        out = frame.reshape(frame.size()[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        out_pose = self.fc_pose(robot_pose)
        out_pose = F.relu(out_pose)
        out = torch.cat((out, out_pose), dim=1)

        val = self.fc_val(out)
        pol = self.fc_pol(out)
        pol = F.softmax(pol, dim=1)
        return val, pol
