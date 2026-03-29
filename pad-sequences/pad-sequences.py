import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs:
        return np.empty((0, max_len or 0))
    L = max_len if max_len is not None else max(len(seq) for seq in seqs)

    N = len(seqs)
    padded_seqs = np.full((N, L), pad_value)

    for i, seq in enumerate(seqs):
        limit = min(len(seq), L)
        padded_seqs[i, :limit] = seq[:limit]

    return padded_seqs