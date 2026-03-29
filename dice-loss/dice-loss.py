import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    p = np.array(p)
    y = np.array(y)
    intersect = np.sum(p*y)
    union = np.sum(p) + np.sum(y)
    dice = (2*intersect+eps)/(union+eps)
    
    return 1-dice
    