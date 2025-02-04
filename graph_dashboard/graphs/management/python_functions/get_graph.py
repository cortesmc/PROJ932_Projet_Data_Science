import argparse, os
import networkx as nx
from scripts.data_processing import load_data, preprocess_data
from scripts.graph_builder import build_graph
from scripts.graph_cleaning import clean_graph
from scripts.clustering import cluster_graph
from scripts.visualization import visualize_graph

def main(json_path, graph_name):
    """
    Main function to execute the entire pipeline.
    """
    # Step 1: Load and preprocess the data
    data = load_data(json_path)
    df = preprocess_data(data)

    # Step 2: Build the graph
    G = build_graph(df)

    # Step 3: Clean the graph
    G = clean_graph(G)

    # Step 4: Apply clustering
    G = cluster_graph(G)

    return G

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process JSON data and generate a graph.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file.")
    parser.add_argument("--graph_name", type=str, required=True, help="Name to save the generated graph.")
    args = parser.parse_args()
    
    # Execute main function with provided arguments
    G = main(args.json_path, args.graph_name)
    
    # Config.
    file_path = args.json_path
    name_graph_saved = args.graph_name

    # Generate folder name based on JSON file name
    json_file_name = os.path.basename(file_path).split('.')[0]
    output_folder = os.path.join( "graphs", "generated_graphs", json_file_name)

    os.makedirs(output_folder, exist_ok=True)
    graph_save_path = os.path.join(output_folder, name_graph_saved + "_0.gexf")

    nx.write_gexf(G, graph_save_path)
