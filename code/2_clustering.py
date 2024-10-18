import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain

def cluster_and_save_graph(input_graph_path, output_graph_path):
    G = nx.read_gexf(input_graph_path)

    partition = community_louvain.best_partition(G)

    nx.set_node_attributes(G, partition, 'community')

    print("Node partitions:")
    for node, community in partition.items():
        print(f"Node: {node}, Community: {community}")

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)

    communities = set(partition.values())
    colors = plt.cm.get_cmap("Set1", len(communities))

    node_colors = [colors(partition[node]) for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    
    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=[d['weight'] for (u, v, d) in edges], alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12)

    plt.title("Clustered Word Co-occurrence Graph")
    plt.axis('off')
    plt.show()

    nx.write_gexf(G, output_graph_path)

if __name__ == "__main__":
    input_graph_path = "cleaned_graph.gexf"
    output_graph_path = "clustered_graph.gexf"
    cluster_and_save_graph(input_graph_path, output_graph_path)
