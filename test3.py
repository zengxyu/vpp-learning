import numpy as np
from scipy.spatial.transform.rotation import Rotation


def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

if __name__ == '__main__':

    # from here it is the code I have written to test this function
    vector1 = np.array([0, 1, 0])
    vector2 = np.array([0, 0, 1])

    rot = rotation_matrix_from_vectors(vector1, vector2)
    rotation_between_vectors = Rotation.from_matrix(rot)

    print("rotation_between_vectors as_quat\n", rotation_between_vectors.as_quat())
    print(np.linalg.norm(rotation_between_vectors.as_quat()))
    print("rotation_between_vectors as_euler\n", rotation_between_vectors.as_euler("xyz", degrees=True))

    # rotated = np.dot(rot, vector1)
    # rotated2 = np.dot(rotation_between_vectors.as_matrix(), vector1)
    #
    # print(rotated)
    # print(rotated2)

    new_rotation = Rotation.from_quat([0.70710678, 0,   0,   0.70710678])

    vector1_rotated_by_rotation1 = np.dot(rotation_between_vectors.as_matrix(), vector2)
    vector1_rotated_by_rotation2 = np.dot(new_rotation.as_matrix(), vector2)

    print("vector1_rotated_by_rotation1:", vector1_rotated_by_rotation1)
    print("vector1_rotated_by_rotation2:", vector1_rotated_by_rotation2)
    direction = np.array([20, 20, 20])
    unit_direction = direction / np.linalg.norm(direction)
    print("unit direciton:", np.linalg.norm(unit_direction))
    x = Rotation.from_quat([0,0,0,1]).as_matrix().transpose()
    print("x:",x)

