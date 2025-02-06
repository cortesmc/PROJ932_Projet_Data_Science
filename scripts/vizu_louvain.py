from pyvis.network import Network

def visualize_graph(G):
    """
    Create an interactive visualization of the graph using PyVis.
    
    Args:
        G (nx.Graph): Input graph to visualize.
    """
    # Create a PyVis network
    net = Network(notebook=True)
    
    # Add nodes
    for node in G.nodes():
        net.add_node(node)
    
    # Add edges with subjects and weights
    for edge in G.edges(data=True):
        node1, node2, data = edge
        subjects = data.get("subjects", {})
        label = ", ".join([f"{subject}: {count}" for subject, count in subjects.items()])
        net.add_edge(node1, node2, weight=data["weight"], title=label)
    
    # Save and display the graph as an HTML file
    net.show("../data/processed/graph_louvain.html")