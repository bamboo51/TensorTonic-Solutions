def hit_rate_at_k(recommendations: list, ground_truth: list, k: int) -> float:
    """
    Returns the fraction of users with a relevant item in their first k recommendations.
    """
    if not recommendations:
        return 0.0
        
    count = 0
    # Write code here
    for index, recommendation in enumerate(recommendations):
        top_k = set(recommendation[:k])
        truth = set(ground_truth[index])

        if top_k.intersection(truth):
            count += 1
    return count / len(recommendations)
            
    
    