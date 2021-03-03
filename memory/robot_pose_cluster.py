from scipy import spatial
from sklearn.cluster import KMeans
import numpy as np


class RobotPoseCluster:

    def __init__(self, k=5, field_size=(256, 256, 256), verbose=True):
        self.k = k
        self.robot_poses = []
        self.centroid = None
        self.field_center = np.array([field_size[0], field_size[1], field_size[2]]) / 2
        self.verbose = verbose

    def add_robot_pose(self, robot_pose):
        self.robot_poses.append(robot_pose)

    def update_cluster(self):
        if self.__len__() > self.k:
            cluster = KMeans(n_clusters=self.k, random_state=9).fit(self.robot_poses)
            self.centroid = cluster.cluster_centers_
        if self.verbose:
            print("Centroids of clusters:{}".format(self.centroid))

    def __len__(self):
        return self.robot_poses.__len__()

    def get_destination(self, robot_pose):
        if self.centroid is None:
            destination = self.field_center
        else:
            dists = spatial.distance.cdist(robot_pose[np.newaxis, :], self.centroid, metric='euclidean')
            centroid_arg = np.argmin(dists, axis=1)
            destination = self.centroid[centroid_arg]
            destination = destination.squeeze()
        return np.array(destination)
