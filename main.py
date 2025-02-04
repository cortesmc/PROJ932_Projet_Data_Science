from scripts.data_processing import load_data, preprocess_data
from scripts.graph_builder import build_graph
from scripts.graph_cleaning import clean_graph
from scripts.clustering import cluster_graph
from scripts.visualization import visualize_graph

def main():
    """
    Main function to execute the entire pipeline.
    """
    # Step 1: Load and preprocess the data
    data = load_data("../data/raw/1eb80fb8b50.json")
    df = preprocess_data(data)

    # Step 2: Build the graph
    G = build_graph(df)

    # Step 3: Clean the graph
    G_cleaned = clean_graph(G)

    # Step 4: Apply clustering
    communities = cluster_graph(G_cleaned)

    # Step 5: Visualize the graph
    visualize_graph(G_cleaned)

if __name__ == "__main__":
    main()