import pfrl
import torch
import torch.nn.functional as F


class NetworkObs(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.frame_con1 = torch.nn.Conv2d(20, 36, kernel_size=4, stride=2, padding=1)
        self.frame_con2 = torch.nn.Conv2d(36, 48, kernel_size=4, stride=2, padding=1)

        self.frame_fc1 = torch.nn.Linear(1728, 512)
        self.frame_fc2 = torch.nn.Linear(512, 256)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc_val = torch.nn.Linear(64, action_size)

    def forward(self, state):
        state = state.float()
        out_frame = F.relu(self.frame_con1(state))
        out_frame = F.relu(self.frame_con2(out_frame))

        out_frame = out_frame.reshape(out_frame.size()[0], -1)
        out_frame = F.relu(self.frame_fc1(out_frame))
        out_frame = F.relu(self.frame_fc2(out_frame))

        out_frame = F.relu(self.fc1(out_frame))
        action_values = self.fc_val(out_frame)

        return pfrl.action_value.DiscreteActionValue(action_values)
