import numpy as np
def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    p_t = np.where(
        targets==1, 
        predictions,
        1-predictions
    )
        
    return np.mean(-alpha*np.pow((1-p_t), gamma)*np.log(p_t))