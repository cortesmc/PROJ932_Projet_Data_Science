from sklearn.decomposition import LatentDirichletAllocation

def extract_topics_with_lda(bow_matrix, n_topics=5):
    """
    Extract topics from segments using LDA.
    
    Args:
        bow_matrix (scipy.sparse.csr_matrix): Bag of Words matrix of segments.
        n_topics (int): Number of topics to extract.
    
    Returns:
        sklearn.decomposition.LatentDirichletAllocation: Fitted LDA model.
        numpy.ndarray: Topic distribution for each segment.
    """
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(bow_matrix)
    topic_distributions = lda.transform(bow_matrix)
    return lda, topic_distributions