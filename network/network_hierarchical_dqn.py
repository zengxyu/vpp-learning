from enum import IntEnum

import torch
from torch import nn
import torch.nn.functional as F


class ControlRole(IntEnum):
    MANAGER = 0
    WORKER = 1


class NetworkManager(torch.nn.Module):
    def __init__(self, manager_output_size=7, action_size=13, robot_pose_size=7):
        super().__init__()
        self.actor_model = NetworkManagerActor()
        self.critic_model = NetworkManagerCritic()

    def policy(self, observed_map, robot_pose):
        return self.actor_model(observed_map, robot_pose)

    def value(self, observed_map, robot_pose, act):
        return self.critic_model(observed_map, robot_pose, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class NetworkManagerActor(torch.nn.Module):
    def __init__(self, manager_output_size=7, robot_pose_size=7):
        super().__init__()
        self.manager_frame_con1 = torch.nn.Conv2d(15, 32, kernel_size=4, stride=2)
        self.manager_frame_con2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.manager_frame_fc1 = torch.nn.Linear(576, 256)
        self.manager_frame_fc2 = torch.nn.Linear(256, 128)

        self.manager_pose_fc1 = torch.nn.Linear(7, 32)
        self.manager_pose_fc2 = torch.nn.Linear(32, 128)

        self.manager_concat_fc = torch.nn.Linear(256, 64)

        self.manager_pose_direction_goal = torch.nn.Linear(64, 3)
        self.manager_rotation_direction_goal = torch.nn.Linear(64, 4)

        self.rrelu = torch.nn.RReLU(0, 1)

    def forward(self, frame, robot_pose):
        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)

        out_manager_frame = F.relu(self.manager_frame_con1(frame))
        out_manager_frame = F.relu(self.manager_frame_con2(out_manager_frame))

        out_manager_frame = out_manager_frame.reshape(out_manager_frame.size()[0], -1)
        out_manager_frame = F.relu(self.manager_frame_fc1(out_manager_frame))
        out_manager_frame = F.relu(self.manager_frame_fc2(out_manager_frame))

        out_manager_pose = F.relu(self.manager_pose_fc1(robot_pose))
        out_manager_pose = F.relu(self.manager_pose_fc2(out_manager_pose))

        out_manager = torch.cat((out_manager_frame, out_manager_pose), dim=1)

        out_manager = F.relu(self.manager_concat_fc(out_manager))

        out_manager_pose_direction_goal = self.manager_pose_direction_goal(out_manager)

        out_manager_rotation_direction_goal = self.manager_rotation_direction_goal(out_manager)

        goal_out = torch.cat((out_manager_pose_direction_goal, out_manager_rotation_direction_goal), dim=1)

        return goal_out


class NetworkManagerCritic(torch.nn.Module):
    def __init__(self, manager_output_size=7, action_size=13, robot_pose_size=7):
        super().__init__()
        self.manager_frame_con1 = torch.nn.Conv2d(15, 32, kernel_size=4, stride=2)
        self.manager_frame_con2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.manager_frame_fc1 = torch.nn.Linear(576, 256)
        self.manager_frame_fc2 = torch.nn.Linear(256, 128)

        self.manager_pose_fc1 = torch.nn.Linear(7, 32)
        # self.manager_pose_fc2 = torch.nn.Linear(32, 128)
        self.manager_goal_fc1 = torch.nn.Linear(7, 32)

        self.manager_concat_fc = torch.nn.Linear(192, 64)

        self.value = torch.nn.Linear(64, 1)

    def forward(self, frame, robot_pose, act):
        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)

        out_manager_frame = F.relu(self.manager_frame_con1(frame))
        out_manager_frame = F.relu(self.manager_frame_con2(out_manager_frame))

        out_manager_frame = out_manager_frame.reshape(out_manager_frame.size()[0], -1)
        out_manager_frame = F.relu(self.manager_frame_fc1(out_manager_frame))
        out_manager_frame = F.relu(self.manager_frame_fc2(out_manager_frame))

        out_manager_pose = F.relu(self.manager_pose_fc1(robot_pose))
        out_manager_goal = F.relu(self.manager_goal_fc1(act))

        out_manager = torch.cat((out_manager_frame, out_manager_pose, out_manager_goal), dim=1)

        out_manager = F.relu(self.manager_concat_fc(out_manager))

        value = self.value(out_manager)

        return value


class NetworkWorker(torch.nn.Module):
    def __init__(self, manager_output_size=7, action_size=13, robot_pose_size=7):
        super().__init__()

        self.worker_fc1a = torch.nn.Linear(3, 32)
        self.worker_fc2a = torch.nn.Linear(32, 64)

        self.worker_fc1b = torch.nn.Linear(3, 32)
        self.worker_fc2b = torch.nn.Linear(32, 64)

        self.worker_fc1c = torch.nn.Linear(3, 32)
        self.worker_fc2c = torch.nn.Linear(32, 64)

        self.worker_fc1d = torch.nn.Linear(4, 32)
        self.worker_fc2d = torch.nn.Linear(32, 64)

        self.worker_fc3 = torch.nn.Linear(256, 128)

        self.worker_fc4 = torch.nn.Linear(128, 32)

        self.worker_fc_val = torch.nn.Linear(32, action_size)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, robot_pose):
        # print("robot_pose[:, 6:] shape:", robot_pose[:, 6:].shape)

        out_worker_a = F.relu(self.worker_fc1a(robot_pose[:, 0:3]))

        out_worker_a = F.relu(self.worker_fc2a(out_worker_a))

        out_worker_b = F.relu(self.worker_fc1b(robot_pose[:, 3:6]))
        out_worker_b = F.relu(self.worker_fc2b(out_worker_b))

        out_worker_c = F.relu(self.worker_fc1c(robot_pose[:, 6:9]))
        out_worker_c = F.relu(self.worker_fc2c(out_worker_c))

        out_worker_d = F.relu(self.worker_fc1d(robot_pose[:, 9:]))
        out_worker_d = F.relu(self.worker_fc2d(out_worker_d))

        out = torch.cat((out_worker_a, out_worker_b, out_worker_c, out_worker_d), dim=1)

        out = F.relu(self.worker_fc3(out))

        out = F.relu(self.worker_fc4(out))

        val = self.worker_fc_val(out)

        return val
