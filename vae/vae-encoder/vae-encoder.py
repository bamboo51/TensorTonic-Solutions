import numpy as np

def vae_encoder(x: np.ndarray, W_mu: np.ndarray, b_mu: np.ndarray, W_logvar: np.ndarray, b_logvar: np.ndarray) -> dict:
    """
    Returns: dict with 'mu' and 'log_var' as np.ndarrays of shape (batch, latent_dim)
    """
    print(x.shape)
    print(W_mu.shape)
    print(b_mu.shape)
    return {"mu": x@W_mu+b_mu, "log_var": x@W_logvar+b_logvar}
