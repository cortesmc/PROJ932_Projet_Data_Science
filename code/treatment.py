import networkx as nx
from pyvis.network import Network

# Charger le fichier GEXF
G = nx.read_gexf("persons.gexf")

# Créer un objet Pyvis Network pour la visualisation
net = Network(notebook=True)

# Convertir le graphe NetworkX en Pyvis
net.from_nx(G)

# Appliquer des options similaires à ForceAtlas2
net.force_atlas_2based()

# Sauvegarder et afficher le graphe
net.show("graph.html")
