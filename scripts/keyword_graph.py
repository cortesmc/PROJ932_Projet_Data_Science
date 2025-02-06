import networkx as nx
from itertools import combinations

def build_keyword_graph(df, segment_subjects):
 
 
    """
    Build a graph of keywords with typed and weighted relations.
    
    Args:
        df (pd.DataFrame): DataFrame containing article data.
        segment_subjects (dict): Dictionary mapping segment indices to subjects.
    
    Returns:
        nx.Graph: Graph of keywords with typed and weighted relations.
    """
    G = nx.Graph()
    
    for segment_idx, row in df.iterrows():
        keywords = row["loc-l"] + row["org-l"] + row["per-l"]
        subject = segment_subjects.get(segment_idx, "unknown")
        
        # Add edges between keywords with the segment's subject
        for kw1, kw2 in combinations(keywords, 2):
            edge_key = (kw1, kw2, subject)
            if G.has_edge(kw1, kw2):
                G[kw1][kw2]["weight"] += 1
                G[kw1][kw2]["subjects"][subject] = G[kw1][kw2]["subjects"].get(subject, 0) + 1
            else:
                G.add_edge(kw1, kw2, weight=1, subjects={subject: 1})
    
    return G