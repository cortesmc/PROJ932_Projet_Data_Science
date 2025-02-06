import networkx as nx
from networkx.algorithms.community import louvain_communities

def cluster_graph(G, min_community_size=10):
    """
    Apply Louvain clustering to detect communities in the graph.
    
    Args:
        G (nx.Graph): Input graph.
        min_community_size (int): Minimum size for communities to keep.
    
    Returns:
        list: List of communities that meet the size threshold.
    """
    # Detect communities using Louvain algorithm
    partition = louvain_communities(G, weight="weight")
    # Filter communities by size
    communities = [community for community in partition if len(community) >= min_community_size]
    return communities

if __name__ == "__main__":
    # Load the cleaned graph
    G = nx.read_gexf("../data/processed/graph_0.gexf")
    # Apply clustering
    communities = cluster_graph(G)
    # Print the number of detected communities
    print(f"Number of detected communities: {len(communities)}")
    # Display community details
    for i, community in enumerate(communities):
        print(f"\n🔹 Community {i+1} ({len(community)} nodes):")
        print(", ".join(community))  # Print node names in this commu