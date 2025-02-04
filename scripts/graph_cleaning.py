import networkx as nx

def clean_graph(G, min_edge_weight=1, min_degree=5):
    """
    Clean the graph by removing edges and nodes that do not meet the criteria.
    
    Args:
        G (nx.Graph): Input graph to clean.
        min_edge_weight (int): Minimum weight for edges to keep.
        min_degree (int): Minimum degree for nodes to keep.
    
    Returns:
        nx.Graph: Cleaned graph.
    """
    # Remove edges with weight below the threshold
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < min_edge_weight]
    G.remove_edges_from(edges_to_remove)
    # Remove nodes with degree below the threshold
    nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree < min_degree]
    G.remove_nodes_from(nodes_to_remove)
    return G

if __name__ == "__main__":
    # Load the graph
    G = nx.read_gexf("../data/processed/graph.gexf")
    # Clean the graph
    G_cleaned = clean_graph(G)
    # Save the cleaned graph to a GEXF file
    nx.write_gexf(G_cleaned, "../data/processed/cleaned_graph.gexf")