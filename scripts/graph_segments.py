import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_segment_graph(df, similarity_threshold=0.2):
    """
    Build a graph where each node is a content segment, and edges are weighted by cosine similarity.

    Args:
        df (pd.DataFrame): DataFrame containing content-segmented data.
        similarity_threshold (float): Minimum similarity score to create an edge.

    Returns:
        nx.Graph: Graph of segments.
    """
    G = nx.Graph()

    # Extract unique content segments
    segments = df["content-segmented"].dropna().unique()

    # Convert segments into vectorized form
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(segments)

    # Compute pairwise similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    for i, segment1 in enumerate(segments):
        G.add_node(segment1, label=segment1)

        for j, segment2 in enumerate(segments):
            if i != j and similarity_matrix[i, j] >= similarity_threshold:
                G.add_edge(segment1, segment2, weight=similarity_matrix[i, j])

    return G

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv("../data/processed/processed_data.csv")

    # Build segment similarity graph
    G_segments = build_segment_graph(df)

    # Save graph
    nx.write_gexf(G_segments, "../data/processed/segment_graph.gexf")
