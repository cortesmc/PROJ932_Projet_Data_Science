import json
import networkx as nx
import matplotlib.pyplot as plt

# Charger les données JSON
with open('1eb80fb8b50.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialiser un graphe
G = nx.Graph()

# Supposons que nous utilisons les co-occurrences de mots-clés dans le même contexte (paragraphe ou jour)
# Ici, j'utilise les données des mots-clés pour construire le graphe
kws = data["metadata"]["all"]["kws"]

# Liste des entités déjà ajoutées
entities = list(kws.keys())

for entity in entities:
    G.add_node(entity)


for i in range(0, len(entities)-1):
    for j in range(i+1, len(entities)):
        # Simuler une co-occurrence en ajoutant des liens entre les entités
        G.add_edge(entities[i], entities[j])

# Visualisation du graphe
nx.draw(G, with_labels=True, node_size=50, font_size=8)
plt.show()

# Exporter le graphe pour Gephi
nx.write_gexf(G, "graph_from_json.gexf")
