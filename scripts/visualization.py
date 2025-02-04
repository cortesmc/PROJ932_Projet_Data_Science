from pyvis.network import Network

def visualize_graph(G):
    """
    Create an interactive visualization of the graph using PyVis.
    
    Args:
        G (nx.Graph): Input graph to visualize.
    """
    # Create a PyVis network
    net = Network(notebook=True)
    # Convert the NetworkX graph to a PyVis graph
    net.from_nx(G)
    # Save and display the graph as an HTML file
    net.show("../data/processed/graph.html")

if __name__ == "__main__":
    import networkx as nx
    # Load the cleaned graph
    G = nx.read_gexf("../data/processed/clustered_graph.gexf")
    # Visualize the graph
    visualize_graph(G)