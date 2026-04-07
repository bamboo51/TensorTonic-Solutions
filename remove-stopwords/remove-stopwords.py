def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    stopwords = set(stopwords)
    return [t for t in tokens if t not in stopwords]