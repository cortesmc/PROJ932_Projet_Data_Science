import networkx as nx
from itertools import combinations

def build_cooccurrence_graph(df):
    """
    Build a co-occurrence graph of keywords from the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing article data.
    
    Returns:
        nx.Graph: Co-occurrence graph of keywords.
    """
    G = nx.Graph()
    
    for _, row in df.iterrows():
        # Combine entities from "loc-l", "org-l", and "per-l"
        keywords = row["loc-l"] + row["org-l"] + row["per-l"]
        
        # Add edges between all pairs of keywords
        for kw1, kw2 in combinations(keywords, 2):
            if G.has_edge(kw1, kw2):
                G[kw1][kw2]["weight"] += 1
            else:
                G.add_edge(kw1, kw2, weight=1)
    
    return G