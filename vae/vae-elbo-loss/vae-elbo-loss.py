import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns: dict with "total", "recon", and "kl" loss values as floats
    """
    # Your implementation here
    kl_loss = -0.5*np.sum(1+log_var-mu**2-np.exp(log_var), axis=1)
    mse_loss = np.sum((x-x_recon)**2, axis=1)

    return {
        "total": (kl_loss+mse_loss).mean(),
        "kl": kl_loss.mean(),
        "recon": mse_loss.mean()
    }
