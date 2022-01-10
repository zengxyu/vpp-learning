import os
import sys
import numpy as np

from torch.utils.data.sampler import Sampler
import torch
import torch.utils.data

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', "capnp"))

try:
    import open3d as o3d
except ImportError:
    raise ImportError(
        "Please install open3d and scipy with `pip install open3d scipy`."
    )


def load_pcd_file(pcd_file_path):
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    points = normalize(pcd.points)

    xyz = np.asarray(points)

    return xyz


def load_mesh_file(mesh_file_path):
    density = 30000
    pcd = o3d.io.read_triangle_mesh(mesh_file_path)
    # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
    pcd.vertices = o3d.utility.Vector3dVector(normalize(pcd.vertices))
    xyz = resample_mesh(pcd, density=density)
    return xyz


def normalize(vertices):
    """
    normalize vertices
    Args:
        vertices:

    Returns:

    """
    # Normalize to a unit cube while preserving aspect ratio
    vertices = np.asarray(vertices)
    vmax = vertices.max(0, keepdims=True)
    vmin = vertices.min(0, keepdims=True)
    normalized_vertices = (vertices - vmin) / (vmax - vmin).max()
    return normalized_vertices


def resample_mesh(mesh_cad, density=1):
    """
    https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.

    param mesh_cad: low-polygon triangle mesh in o3d.geometry.TriangleMesh
    param density: density of the point cloud per unit area
    param return_numpy: return numpy format or open3d pointcloud format
    return resampled point cloud

    Reference :
      [1] Barycentric coordinate system
      \begin{align}
        P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
      \end{align}
    """
    faces = np.array(mesh_cad.triangles).astype(int)
    vertices = np.array(mesh_cad.vertices)

    vec_cross = np.cross(
        vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
        vertices[faces[:, 1], :] - vertices[faces[:, 2], :],
    )
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))

    n_samples = (np.sum(face_areas) * density).astype(int)
    # face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Bug fix by Yangyan (yangyan.lee@gmail.com)
    n_samples_per_face = np.ceil(density * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc: acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]

    P = (
            (1 - np.sqrt(r[:, 0:1])) * A
            + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B
            + np.sqrt(r[:, 0:1]) * r[:, 1:] * C
    )

    return P


def PointCloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.)
    return pcd


class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)
