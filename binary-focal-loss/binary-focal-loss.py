import numpy as np
def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    predictions = np.array(predictions)
    targets = np.array(targets)

    eps = 1e-9
    predictions = np.clip(predictions, eps, 1 - eps)
    
    p_t = np.where(
        targets==1, 
        predictions,
        1-predictions
    )
    # in real case: alpha depends on the class:
    # for positive class → alpha
    # for negative class → 1 - alpha    
    return np.mean(-alpha*np.power((1-p_t), gamma)*np.log(p_t))