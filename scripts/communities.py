import community as community_louvain

def detect_communities(G):
    """
    Detect communities in the graph using the Louvain method.
    
    Args:
        G (nx.Graph): Input graph.
    
    Returns:
        dict: A dictionary where keys are nodes and values are community IDs.
    """
    partition = community_louvain.best_partition(G)
    return partition