import sys
import os
import time
import zmq
import capnp
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "capnp"))
sys.path.append('/home/wshi/workspace/vpp-learning/capnp')
print(sys.path)

import action_capnp
import observation_capnp
import numpy as np
import roslaunch


class EnvironmentClient:
    def __init__(self, handle_simulation=False):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.bind("tcp://*:5555")

        if handle_simulation:
            self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(self.uuid)
            self.launch_file = roslaunch.rlutil.resolve_launch_arguments(['vpp_learning_ros', 'ur_with_cam.launch'])
            # self.launch_args = ['world_name:=world14_hc', 'base:=retractable', 'gui:=false']
            self.launch_args = ['world_name:=world19', 'base:=retractable', 'gui:=false']

            self.launch_files = [(self.launch_file[0], self.launch_args)]

    def startSimulation(self):
        self.parent = roslaunch.parent.ROSLaunchParent(self.uuid, self.launch_files)
        self.parent.start()

    def spinSimulationEventLoop(self):
        self.parent.spin_once()

    def shutdownSimulation(self):
        print('Shutting down')
        self.parent.shutdown()
        print('Shutdown complete')

    def decodeObservation(self, obs_msg):
        shape = (obs_msg.layers, obs_msg.height, obs_msg.width)
        unknownCount = np.reshape(np.array(obs_msg.unknownCount), shape)
        freeCount = np.reshape(np.array(obs_msg.freeCount), shape)
        occupiedCount = np.reshape(np.array(obs_msg.occupiedCount), shape)
        roiCount = np.reshape(np.array(obs_msg.roiCount), shape)

        robotPose = self.poseToNumpyArray(obs_msg.robotPose)
        robotJoints = np.array(obs_msg.robotJoints)

        if obs_msg.planningTime > 0:
            reward = obs_msg.foundRois
        else:
            reward = 0

        return unknownCount, freeCount, occupiedCount, roiCount, robotPose, robotJoints, reward

    def poseToNumpyArray(self, pose):
        return np.array([pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y,
                         pose.orientation.z, pose.orientation.w])

    def encodeGoalPose(self, action_msg, data):
        action_msg.init("goalPose")
        action_msg.goalPose.position.x = data[0]
        action_msg.goalPose.position.y = data[1]
        action_msg.goalPose.position.z = data[2]
        action_msg.goalPose.orientation.x = data[3]
        action_msg.goalPose.orientation.y = data[4]
        action_msg.goalPose.orientation.z = data[5]
        action_msg.goalPose.orientation.w = data[6]

    def encodeRelativePose(self, action_msg, data):
        action_msg.init("relativePose")
        action_msg.relativePose.position.x = data[0]
        action_msg.relativePose.position.y = data[1]
        action_msg.relativePose.position.z = data[2]
        action_msg.relativePose.orientation.x = data[3]
        action_msg.relativePose.orientation.y = data[4]
        action_msg.relativePose.orientation.z = data[5]
        action_msg.relativePose.orientation.w = data[6]

    def encodeRandomizationParameters(self, action_msg, min_point, max_point, min_dist):
        action_msg.init("resetAndRandomize")
        action_msg.resetAndRandomize.min.x = min_point[0]
        action_msg.resetAndRandomize.min.y = min_point[1]
        action_msg.resetAndRandomize.min.z = min_point[2]
        action_msg.resetAndRandomize.max.x = max_point[0]
        action_msg.resetAndRandomize.max.y = max_point[1]
        action_msg.resetAndRandomize.max.z = max_point[2]
        action_msg.resetAndRandomize.minDist = min_dist

    def sendAction(self, action_msg):
        self.socket.send(action_msg.to_bytes())

        #  Get the reply.
        message = self.socket.recv()
        obs_msg = observation_capnp.Observation.from_bytes(message)
        return self.decodeObservation(obs_msg)

    def sendRelativeJointTarget(self, joint_values):
        action_msg = action_capnp.Action.new_message()
        action_msg.relativeJointTarget = joint_values
        return self.sendAction(action_msg)

    def sendAbsoluteJointTarget(self, joint_values):
        action_msg = action_capnp.Action.new_message()
        action_msg.absoluteJointTarget = joint_values
        return self.sendAction(action_msg)

    def sendGoalPose(self, goal_pose):
        action_msg = action_capnp.Action.new_message()
        self.encodeGoalPose(action_msg, goal_pose)
        return self.sendAction(action_msg)

    def sendRelativePose(self, relative_pose):
        action_msg = action_capnp.Action.new_message()
        self.encodeRelativePose(action_msg, relative_pose)
        return self.sendAction(action_msg)

    def sendReset(self):
        action_msg = action_capnp.Action.new_message()
        action_msg.reset = None
        return self.sendAction(action_msg)

    def sendResetAndRandomize(self, min_point, max_point, min_dist):
        action_msg = action_capnp.Action.new_message()
        self.encodeRandomizationParameters(action_msg, min_point, max_point, min_dist)
        return self.sendAction(action_msg)


def main(args):
    # client = EnvironmentClient()
    # unknownCount, freeCount, occupiedCount, roiCount, robotPose, robotJoints, reward = client.sendRelativeJointTarget([-0.1, 0, 0, 0, 0, 0]) # client.sendRelativePose([0.1, 0, 0, 0, 0, 0, 1])
    # print("unknownCount")
    # print(unknownCount)
    # print("freeCount")
    # print(freeCount)
    # print("occupiedCount")
    # print(occupiedCount)
    # print("roiCount")
    # print(roiCount)
    # print("robotPose")
    # print(robotPose)
    # print("robotJoints")
    # print(robotJoints)
    # print("Reward", reward)
    # client.sendResetAndRandomize([-1, -1, -0.1], [1, 1, 0.1], 0.4)
    client = EnvironmentClient(handle_simulation=True)

    for i in range(200):
        print("============================================================eps {}".format(i))
        client.startSimulation()
        print('SEND_RESET_1')
        observation = client.sendResetAndRandomize([-1, -1, -0.1], [1, 1, 0.1], 0.4)

        print('SEND_MOVE1_1')
        observation = client.sendRelativeJointTarget([-0.1, 0, 0, 0, 0, 0])
        print('SEND_MOVE2_1')
        observation = client.sendRelativeJointTarget([0, 0.1, 0, 0, 0, 0])
        print('SHUTDOWN_1')
        client.shutdownSimulation()
        print('RESTART')
        client.startSimulation()
        print('SEND_RESET_2')
        observation = client.sendResetAndRandomize([-1, -1, -0.1], [1, 1, 0.1], 0.4)
        print('SEND_MOVE1_2')
        observation = client.sendRelativeJointTarget([-0.1, 0, 0, 0, 0, 0])
        print('SEND_MOVE2_2')
        observation = client.sendRelativeJointTarget([0, 0.1, 0, 0, 0, 0])
        print('SHUTDOWN_2')
        client.shutdownSimulation()


if __name__ == '__main__':
    main(sys.argv)
