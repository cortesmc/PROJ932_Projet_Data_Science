def assign_subjects_to_segments(topic_distributions):
    """
    Assign a subject to each segment based on the most probable topic.
    
    Args:
        topic_distributions (numpy.ndarray): Topic distribution for each segment.
    
    Returns:
        dict: A dictionary where keys are segment indices and values are topic IDs.
    """
    segment_subjects = {}
    for segment_idx, topic_dist in enumerate(topic_distributions):
        topic_id = topic_dist.argmax()  # Assign the most probable topic
        segment_subjects[segment_idx] = topic_id
    return segment_subjects