import sys
import signal
import os
import time
import zmq
import capnp

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "capnp"))
import action_capnp
import observation_capnp
import numpy as np
import roslaunch
from timeit import default_timer as timer


class EnvironmentClient:
    def __init__(self, handle_simulation=False):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.RCVTIMEO = 1000  # in milliseconds
        self.socket.bind("tcp://*:5555")

        if handle_simulation:
            self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(self.uuid)
            self.launch_file = roslaunch.rlutil.resolve_launch_arguments(['vpp_learning_ros', 'ur_with_cam.launch'])
            self.launch_args = ['world_name:=world19', 'base:=retractable']
            self.launch_files = [(self.launch_file[0], self.launch_args)]
            signal.signal(signal.SIGINT, self.sigint_handler)

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
        robotPose = self.poseToNumpyArray(obs_msg.robotPose)
        robotJoints = np.array(obs_msg.robotJoints)

        if obs_msg.planningTime > 0:
            # reward = obs_msg.foundRois / obs_msg.planningTime
            reward = obs_msg.foundRois
        else:
            reward = 0

        which = obs_msg.map.which()
        if which == 'countMap':
            shape = (obs_msg.map.countMap.layers, obs_msg.map.countMap.height, obs_msg.map.countMap.width)
            unknownCount = np.reshape(np.array(obs_msg.map.countMap.unknownCount), shape)
            freeCount = np.reshape(np.array(obs_msg.map.countMap.freeCount), shape)
            occupiedCount = np.reshape(np.array(obs_msg.map.countMap.occupiedCount), shape)
            roiCount = np.reshape(np.array(obs_msg.map.countMap.roiCount), shape)
            return unknownCount, freeCount, occupiedCount, roiCount, robotPose, robotJoints, reward
        elif which == 'pointcloud':
            print('Converting pointcloud...')
            start = timer()
            points = np.asarray(obs_msg.map.pointcloud.points)
            labels = np.asarray(obs_msg.map.pointcloud.voxelgrid)
            points = points.reshape((len(labels), 3))
            end = timer()
            print('Converting pointcloud took', end - start, 's')
            return points, labels, robotPose, robotJoints, reward

        elif which == "voxelgrid":
            print('Converting voxelgrid...')
            start = timer()
            # points = np.asarray(obs_msg.map.pointcloud.points)
            # labels = np.asarray(obs_msg.map.voxelgrid.labels)
            voxelgrid = obs_msg.map.voxelgrid
            # points = points.reshape((len(labels), 3))
            end = timer()
            print('Converting voxelgrid took', end - start, 's')
            return voxelgrid, robotPose, robotJoints, reward

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

    def sendAction(self, action_msg):
        while True:
            print('Sending message')
            try:
                self.socket.send(action_msg.to_bytes(), flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                print('Could not send message, trying again in 1s...')
                time.sleep(1)
                continue
            break

        while True:
            #  Get the reply.
            print('Receiving message')
            try:
                message = self.socket.recv()
            except zmq.ZMQError:
                print('No response received, trying again...')
                continue
            break
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

    def sendReset(self, randomize=False, min_point=[-1, -1, -0.1], max_point=[1, 1, 0.1], min_dist=0.4,
                  map_type='unchanged'):
        action_msg = action_capnp.Action.new_message()
        action_msg.init("reset")
        if randomize:
            action_msg.reset.randomize = True
            action_msg.reset.randomizationParameters.min.x = min_point[0]
            action_msg.reset.randomizationParameters.min.y = min_point[1]
            action_msg.reset.randomizationParameters.min.z = min_point[2]
            action_msg.reset.randomizationParameters.max.x = max_point[0]
            action_msg.reset.randomizationParameters.max.y = max_point[1]
            action_msg.reset.randomizationParameters.max.z = max_point[2]
            action_msg.reset.randomizationParameters.minDist = min_dist

        action_msg.reset.mapType = map_type
        return self.sendAction(action_msg)

    def sigint_handler(self, sig, frame):
        print('SIGINT received')
        self.shutdownSimulation()
        self.socket.close()
        sys.exit(0)


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
    # client.sendReset(randomize=True, min_point=[-1, -1, -0.1], max_point=[1, 1, 0.1], min_dist=0.4)

    client = EnvironmentClient(handle_simulation=True)
    client.startSimulation()
    print('SEND_RESET_1')
    observation = client.sendReset(randomize=True, min_point=[-1, -1, -0.1], max_point=[1, 1, 0.1], min_dist=0.4)
    print('SEND_MOVE1_1')
    observation = client.sendRelativeJointTarget([-0.1, 0, 0, 0, 0, 0])
    print('SEND_MOVE2_1')
    observation = client.sendRelativeJointTarget([0, 0.1, 0, 0, 0, 0])
    print('SHUTDOWN_1')
    client.shutdownSimulation()
    print('RESTART')
    client.startSimulation()
    print('SEND_RESET_2')
    observation = client.sendReset(randomize=True, min_point=[-1, -1, -0.1], max_point=[1, 1, 0.1], min_dist=0.4)
    print('SEND_MOVE1_2')
    observation = client.sendRelativeJointTarget([-0.1, 0, 0, 0, 0, 0])
    print('SEND_MOVE2_2')
    observation = client.sendRelativeJointTarget([0, 0.1, 0, 0, 0, 0])
    print('SHUTDOWN_2')
    client.shutdownSimulation()


if __name__ == '__main__':
    main(sys.argv)
