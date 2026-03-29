import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.array(y)

    values, counts = np.unique(y, return_counts=True)
    probs = counts/counts.sum()
    eps=1e-8
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs*np.log2(probs))