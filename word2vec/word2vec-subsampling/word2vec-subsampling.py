import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    counts = counts.float()

    f =  counts / counts.sum()
    keep_probs = torch.sqrt(t/f)
    return torch.clamp(keep_probs, max=1.0)
