import torch
import torch.nn.functional as F


class PPOPolicy3DUnknownMap(torch.nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.fc1 = torch.nn.Linear(324, 64)
        self.fc2 = torch.nn.Linear(64, 32)

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
