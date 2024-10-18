import networkx as nx
import matplotlib.pyplot as plt

def clean_and_save_graph(input_graph_path, output_graph_path, min_edge_weight, min_degree):
    G = nx.read_gexf(input_graph_path)

    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < min_edge_weight]
    G.remove_edges_from(edges_to_remove)

    nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree < min_degree]
    G.remove_nodes_from(nodes_to_remove)

    pos = nx.spring_layout(G, k=0.1, iterations=50)

    plt.figure(figsize=(12, 8))
    
    node_colors = ['#1f78b4' for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)

    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=[d['weight'] for (u, v, d) in edges], alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12)

    plt.title("Cleaned and Spring Layout Graph")
    plt.axis('off')
    plt.show()

    nx.write_gexf(G, output_graph_path)

if __name__ == "__main__":
    input_graph_path = "graph_0.gexf"
    output_graph_path = "cleaned_graph.gexf" 
    min_edge_weight = 2
    min_degree = 2
    clean_and_save_graph(input_graph_path, output_graph_path, min_edge_weight, min_degree)
