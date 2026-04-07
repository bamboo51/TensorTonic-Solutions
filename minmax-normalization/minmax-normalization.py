import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    X = np.array(X, dtype=float)
    min_val = np.min(X, axis=axis, keepdims=True)
    max_val = np.max(X, axis=axis, keepdims=True)
    
    denom = max_val - min_val
    denom = np.where(denom < eps, 1.0, denom)  # avoid division by zero
    
    norm_X = (X - min_val) / denom
    return norm_X