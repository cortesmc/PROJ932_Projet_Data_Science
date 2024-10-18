import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import louvain_communities

def cluster_and_save_graph(input_graph_path, output_graph_path, display_graph, save_graph, min_community_size=10):
    G = nx.read_gexf(input_graph_path)

    partition = louvain_communities(G, weight='weight', max_level = 3, resolution=1.0, seed=42)

    community_mapping = {}
    for idx, community in enumerate(partition):
        for node in community:
            community_mapping[node] = idx

    nx.set_node_attributes(G, community_mapping, 'community')

    filtered_nodes = []
    for community in partition:
        if len(community) >= min_community_size:
            filtered_nodes.extend(community)

    G_filtered = G.subgraph(filtered_nodes).copy()

    print(f"Filtered graph contains {len(G_filtered.nodes())} nodes and {len(G_filtered.edges())} edges.")

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G_filtered)

    communities = set(community_mapping[node] for node in G_filtered.nodes())
    colors = plt.cm.get_cmap("Set1", len(communities))

    node_colors = [colors(community_mapping[node]) for node in G_filtered.nodes()]
    nx.draw_networkx_nodes(G_filtered, pos, node_color=node_colors, node_size=700)

    edges = G_filtered.edges(data=True)
    nx.draw_networkx_edges(G_filtered, pos, edgelist=edges, width=[d['weight'] for (u, v, d) in edges], alpha=0.5)

    nx.draw_networkx_labels(G_filtered, pos, font_size=12)

    plt.title("Louvain Clustered Graph (Large Communities)")
    plt.axis('off')

    if display_graph:
        plt.show()

    if save_graph:
        nx.write_gexf(G_filtered, output_graph_path)

if __name__ == "__main__":
    input_graph_path = "cleaned_graph.gexf" 
    output_graph_path = "louvain_filtered_graph.gexf" 
    cluster_and_save_graph(input_graph_path, output_graph_path, display_graph=True, save_graph=True, min_community_size=10)
