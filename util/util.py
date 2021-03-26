import numpy as np


def get_eu_distance(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))
