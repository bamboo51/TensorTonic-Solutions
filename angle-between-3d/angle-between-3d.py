import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    v = np.array(v)
    w = np.array(w)
    norm_v = np.linalg.norm(v, axis=-1)
    norm_w = np.linalg.norm(w, axis=-1)
    cos_theta = np.dot(v, w)/(norm_v*norm_w)
    return np.arccos(cos_theta)