def display_topics(lda, feature_names, n_top_words=10):
    """
    Display the top words for each topic.
    
    Args:
        lda (sklearn.decomposition.LatentDirichletAllocation): Fitted LDA model.
        feature_names (list): List of feature names (words).
        n_top_words (int): Number of top words to display for each topic.
    """
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"Topic {topic_idx}: {', '.join(top_words)}")