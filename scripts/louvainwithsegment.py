import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms.community import louvain_communities

def build_segment_graph(df, threshold=0.2):
    """
    Build a graph of segments where edges represent high cosine similarity.
    
    Args:
        df (pd.DataFrame): DataFrame containing article data with "content-segmented".
        threshold (float): Similarity threshold to create edges.
    
    Returns:
        nx.Graph: Graph where nodes are segments and edges indicate high similarity.
    """
    # Extract segments
    segments = df["content-segmented"].dropna().unique()  # Remove NaN and duplicate segments
    
    # Convert text segments into TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(segments)

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Build the graph
    G = nx.Graph()
    
    for i, segment in enumerate(segments):
        G.add_node(segment, label=f"Segment {i}")

    # Create edges based on similarity threshold
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            similarity = similarity_matrix[i, j]
            if similarity >= threshold:  # Only connect if similarity is high
                G.add_edge(segments[i], segments[j], weight=similarity)

    return G

def cluster_segments(G):
    """
    Apply Louvain clustering to segment graph.
    
    Args:
        G (nx.Graph): Graph with segments as nodes.
    
    Returns:
        dict: Mapping of segments to cluster IDs.
    """
    partition = louvain_communities(G, weight="weight")
    
    # Create a mapping of segment -> cluster_id
    segment_clusters = {}
    for cluster_id, community in enumerate(partition):
        for segment in community:
            segment_clusters[segment] = cluster_id
    
    return segment_clusters

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv("../data/processed/processed_data.csv")
    
    # Step 1: Build the segment similarity graph
    segment_graph = build_segment_graph(df)

    # Step 2: Apply Louvain clustering
    segment_clusters = cluster_segments(segment_graph)

    # Step 3: Save the clustering results
    df["segment_cluster"] = df["content-segmented"].map(segment_clusters)
    df.to_csv("../data/processed/clustered_segments.csv", index=False)

    print(f"✅ Louvain clustering completed! Clusters saved in: ../data/processed/clustered_segments.csv")
