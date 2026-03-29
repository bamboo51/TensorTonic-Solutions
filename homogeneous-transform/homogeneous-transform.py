import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.array(T)
    points = np.array(points)
    is_single_point = points.ndim == 1
    if is_single_point:
        points = points.reshape(1, 3)
    N = points.shape[0]
    ones = np.ones((N, 1))
    points_homo = np.hstack([points, ones])

    transformed_pts_homo = points_homo @ T.T
    w = transformed_pts_homo[:, 3:4]

    transformed_pts = transformed_pts_homo[:, :3] / w
    if is_single_point:
        return transformed_pts.flatten()

    return transformed_pts