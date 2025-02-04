import networkx as nx
import pandas as pd
from itertools import combinations
from utils import ensure_hashable

def build_graph(df):
    """
    Build a graph from the processed DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing article data.
    
    Returns:
        nx.Graph: Graph built from the entities in the DataFrame.
    """
    G = nx.Graph()
    for _, row in df.iterrows():
        # Combine entities from "loc-l", "org-l", and "per-l"
        entities = row["loc-l"] + row["org-l"] + row["per-l"]
        # Create edges between all pairs of entities
        for entity1, entity2 in combinations(entities, 2):
            # Ensure entities are hashable
            entity1 = ensure_hashable(entity1)
            entity2 = ensure_hashable(entity2)
            # Add or update the edge in the graph
            if G.has_edge(entity1, entity2):
                G[entity1][entity2]["weight"] += 1
            else:
                G.add_edge(entity1, entity2, weight=1)
    return G

if __name__ == "__main__":
    # Load the processed data
    df = pd.read_csv("../data/processed/processed_data.csv")
    # Build the graph
    G = build_graph(df)
    # Save the graph to a GEXF file
    nx.write_gexf(G, "../data/processed/graph.gexf")