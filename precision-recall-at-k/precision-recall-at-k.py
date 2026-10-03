def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    recommended = set(recommended[:k])
    relevant = set(relevant)
    intersect = recommended.intersection(relevant)
    print(intersect)
    return [len(intersect)/k, len(intersect)/len(relevant)]