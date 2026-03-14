import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Convert inputs to numpy arrays to ensure proper operations
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    s = np.array(s, dtype=float)
    
    # Update cache: s_new = β * s + (1-β) * g²
    s_new = beta * s + (1 - beta) * (g ** 2)
    
    # Update weights: w_new = w - lr * g / (√s_new + eps)
    w_new = w - lr * g / (np.sqrt(s_new) + eps)
    
    return w_new, s_new
    