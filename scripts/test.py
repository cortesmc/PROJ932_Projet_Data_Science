import networkx as nx

# Chargez le graphe depuis le fichier GEXF
G = nx.read_gexf("../data/processed/clustered_graph.gexf")

# Vérifiez les nœuds et les arêtes
print("Nombre de nœuds :", G.number_of_nodes())
print("Nombre d'arêtes :", G.number_of_edges())

# Affichez les premiers nœuds et arêtes
print("Premiers nœuds :", list(G.nodes)[:5])
print("Premières arêtes :", list(G.edges)[:5])