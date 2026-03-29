import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    err = y_true - y_pred
    abs_err = np.abs(err)
    
    loss = np.where(
        abs_err <= delta,
        0.5 * err**2,
        delta * (abs_err - 0.5 * delta)
    )
    
    return np.mean(loss)
        