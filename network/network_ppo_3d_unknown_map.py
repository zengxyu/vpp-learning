import torch
import torch.nn.functional as F


class PPOPolicy3DUnknownMap3(torch.nn.Module):
    def __init__(self, action_size=13, robot_pose_size=7):
        super().__init__()

        self.pose_fc1 = torch.nn.Linear(robot_pose_size, 64)
        self.pose_fc2 = torch.nn.Linear(64, 32)

        self.fc_val = torch.nn.Linear(32, 1)
        self.fc_pol = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, robot_pose):
        out_pose = F.relu(self.pose_fc1(robot_pose))
        out_pose = F.relu(self.pose_fc2(out_pose))

        val = self.fc_val(out_pose)
        pol = self.fc_pol(out_pose)
        pol = F.softmax(pol, dim=1)
        return val, pol


class PPOPolicy3DUnknownMap4(torch.nn.Module):
    def __init__(self, action_size=13, robot_pose_size=7):
        super().__init__()

        self.pose_fc1 = torch.nn.Linear(robot_pose_size, 64)
        self.pose_fc2 = torch.nn.Linear(64, 128)
        self.pose_fc3 = torch.nn.Linear(128, 32)

        self.fc_val = torch.nn.Linear(32, 1)
        self.fc_pol = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, robot_pose):
        out_pose = F.relu(self.pose_fc1(robot_pose))
        out_pose = F.relu(self.pose_fc2(out_pose))
        out_pose = F.relu(self.pose_fc3(out_pose))

        val = self.fc_val(out_pose)
        pol = self.fc_pol(out_pose)
        pol = F.softmax(pol, dim=1)
        return val, pol


class PPOPolicy3DUnknownMap5(torch.nn.Module):
    def __init__(self, action_size=13, robot_pose_size=7):
        super().__init__()

        self.pose_fc1a = torch.nn.Linear(3, 32)
        self.pose_fc2a = torch.nn.Linear(32, 64)

        self.pose_fc1b = torch.nn.Linear(3, 32)
        self.pose_fc2b = torch.nn.Linear(32, 64)

        self.pose_fc1c = torch.nn.Linear(3, 32)
        self.pose_fc2c = torch.nn.Linear(32, 64)

        self.pose_fc1d = torch.nn.Linear(4, 32)
        self.pose_fc2d = torch.nn.Linear(32, 64)

        self.pose_fc3 = torch.nn.Linear(256, 128)

        self.pose_fc4 = torch.nn.Linear(128, 32)

        self.fc_val = torch.nn.Linear(32, 1)
        self.fc_pol = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, robot_pose):
        out_pose_a = F.relu(self.pose_fc1a(robot_pose[:, 0:3]))
        out_pose_a = F.relu(self.pose_fc2a(out_pose_a))

        out_pose_b = F.relu(self.pose_fc1b(robot_pose[:, 3:6]))
        out_pose_b = F.relu(self.pose_fc2b(out_pose_b))

        out_pose_c = F.relu(self.pose_fc1c(robot_pose[:, 6:9]))
        out_pose_c = F.relu(self.pose_fc2c(out_pose_c))

        out_pose_d = F.relu(self.pose_fc1d(robot_pose[:, 9:]))
        out_pose_d = F.relu(self.pose_fc2d(out_pose_d))

        out = torch.cat((out_pose_a, out_pose_b, out_pose_c, out_pose_d), dim=1)

        out = F.relu(self.pose_fc3(out))

        out = F.relu(self.pose_fc4(out))

        val = self.fc_val(out)
        pol = self.fc_pol(out)
        pol = F.softmax(pol, dim=1)
        return val, pol


class PPOPolicy3DUnknownMap2(torch.nn.Module):
    def __init__(self, action_size=13, robot_pose_size=7):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(3, 16, kernel_size=4, stride=2)
        self.frame_con2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.frame_fc1 = torch.nn.Linear(288, 512)
        self.frame_fc2 = torch.nn.Linear(512, 128)

        self.pose_fc1 = torch.nn.Linear(robot_pose_size, 32)
        self.pose_fc2 = torch.nn.Linear(32, 128)

        self.concat_fc = torch.nn.Linear(256, 64)

        self.fc_val = torch.nn.Linear(64, 1)
        self.fc_pol = torch.nn.Linear(64, action_size)

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
