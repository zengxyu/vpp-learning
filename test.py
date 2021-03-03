import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

config_dir = "config_dir"

if __name__ == '__main__':
    # with open(os.path.join(config_dir, "observed_map_mean_std.pkl"), 'rb') as f:
    #     observed_map_mean, observed_map_std = pickle.load(f)
    #
    # with open(os.path.join(config_dir, "robot_pose_mean_std.pkl"), 'rb') as f:
    #     robot_pose_mean, robot_pose_std = pickle.load(f)
    #     print("robot_pose_mean:{}".format(robot_pose_mean))
    #
    # with open(os.path.join(config_dir, "reward_mean_std.pkl"), 'rb') as f:
    #     reward_mean, reward_std = pickle.load(f)
    #     print("reward_mean:{}".format(reward_mean))
    r = Rotation.from_quat([0.69840112, -0.69840112, -0.11061587, 0.11061587])
    d = r.as_quat()
    e = r.as_euler('xyz', degrees=True)
    print("quaterinion:", d)
    print("euler:", e)

    r = Rotation.from_quat([-0.11061587,  0.69840112, -0.69840112,  0.11061587])
    d = r.as_quat()
    e = r.as_euler('xyz', degrees=True)
    print("quaterinion:", d)
    print("euler:", e)

    r2 = Rotation.from_euler('xyz', (90, 180, 18), degrees=True)
    q = r2.as_quat()
    print("quaterinion:", q)

    r3 = Rotation.from_euler('xyz', (-90, 0, -162), degrees=True)
    Rotation.from_rotvec()
    q = r3.as_quat()
    print("quaterinion:", q)



