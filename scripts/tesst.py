import networkx as nx

# Chargez le graphe nettoyé
G_cleaned = nx.read_gexf("../data/processed/cleaned_graph.gexf")

# Vérifiez les nœuds et les arêtes
print("Nombre de nœuds dans cleaned_graph :", G_cleaned.number_of_nodes())
print("Nombre d'arêtes dans cleaned_graph :", G_cleaned.number_of_edges())

# Affichez les premiers nœuds et arêtes
print("Premiers nœuds dans cleaned_graph :", list(G_cleaned.nodes)[:60])
print("Premières arêtes dans cleaned_graph :", list(G_cleaned.edges)[:50])