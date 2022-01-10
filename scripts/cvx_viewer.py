import sys
import os
import numpy as np
import capnp
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "capnp"))
import voxelgrid_capnp

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
    def __init__(self, switch_geoms, fixed_geoms):
        self.switch_geoms = switch_geoms
        self.fixed_geoms = fixed_geoms
        self.cur_ind = 0

    def switch_geom(self, vis):
        self.cur_ind = (self.cur_ind + 1) % len(self.switch_geoms)
        vis.clear_geometries()
        vis.add_geometry(self.switch_geoms[self.cur_ind], False)
        for geom in self.fixed_geoms:
            vis.add_geometry(geom)


if len(sys.argv) < 2:
    print('File path not specified')
    sys.exit()

with open(sys.argv[1]) as file:
    voxelgrid = voxelgrid_capnp.Voxelgrid.read(file, traversal_limit_in_words=2**32)
    labels = np.asarray(voxelgrid.voxelgrid)
    shape = tuple(voxelgrid.shape)
    # labels = labels.reshape(shape)
    center = np.asarray(voxelgrid.center)
    resolution = voxelgrid.resolution

    indices = np.moveaxis(np.indices(shape), 0, -1)
    points = indices * resolution
    nat_center_point = points[tuple(int(s/2) for s in shape)]
    lin_shape = np.prod(shape)
    points = points.reshape(lin_shape, len(shape))
    offset = center - nat_center_point
    points = points + offset

    points_all = points[np.where(labels != 3)]
    points_occ = points[np.where((labels == 1) | (labels == 2))]
    points_roi = points[np.where(labels == 2)]
    pcd_all = PointCloud(points_all)
    pcd_occ = PointCloud(points_occ)
    pcd_roi = PointCloud(points_roi)

    center_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2, center)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd_all)
    vis.add_geometry(center_coord)
    ghand = GeometryHandler([pcd_all, pcd_occ, pcd_roi], [center_coord])
    glfw_key_space = 32
    vis.register_key_callback(glfw_key_space, ghand.switch_geom)
    vis.run()
