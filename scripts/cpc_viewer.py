import sys
import os
import numpy as np
import capnp
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "capnp"))
import pointcloud_capnp

try:
    import open3d as o3d
except ImportError:
    raise ImportError(
        "Please install open3d and scipy with `pip install open3d scipy`."
    )


def PointCloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


class GeometryHandler:
    def __init__(self, geoms):
        self.geoms = geoms
        self.cur_ind = 0

    def switch_geom(self, vis):
        self.cur_ind = (self.cur_ind + 1) % len(self.geoms)
        vis.clear_geometries()
        vis.add_geometry(self.geoms[self.cur_ind], False)


if len(sys.argv) < 2:
    print('File path not specified')
    sys.exit()

legacy_mode = False
if len(sys.argv) > 2 and sys.argv[2] == '--legacy':
    legacy_mode = True
    import pointcloud_old_capnp

with open(sys.argv[1]) as file:
    if legacy_mode:
        pointcloud = pointcloud_old_capnp.PointcloudOld.read(file, traversal_limit_in_words=2**32)
    else:
        pointcloud = pointcloud_capnp.Pointcloud.read(file, traversal_limit_in_words=2**32)
    points = np.asarray(pointcloud.points)
    labels = np.asarray(pointcloud.voxelgrid)
    points = points.reshape((len(labels), 3))
    points_occ = points[np.where(labels != 0)]
    points_roi = points[np.where(labels == 2)]
    pcd = PointCloud(points)
    pcd_occ = PointCloud(points_occ)
    pcd_roi = PointCloud(points_roi)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)
    ghand = GeometryHandler([pcd, pcd_occ, pcd_roi])
    glfw_key_space = 32
    vis.register_key_callback(glfw_key_space, ghand.switch_geom)
    vis.run()
