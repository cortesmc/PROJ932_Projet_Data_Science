import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from urllib.parse import urlparse
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import argparse
import os

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process JSON data and generate a graph.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file.")
    parser.add_argument("--graph_name", type=str, required=True, help="Name to save the generated graph.")
    args = parser.parse_args()

    # Config.
    file_path = args.json_path
    name_graph_saved = args.graph_name

    segment_key = ["kws-l", "loc-l", "org-l", "per-l"]
    Years = ["2024"]

    columns_to_use = ["loc-l", "org-l", "per-l"]
    ignore_words = ["Telegram", "TikTok", "Sputnik Africa"]

    save_graph = True

    colors = ['#1f78b4',
              '#33a02c',
              '#e31a1c',
              '#ff7f00',
              '#6a3d9a']

    # Generate folder name based on JSON file name
    output_folder = os.path.join( "static", "gexf")

    if save_graph:
        os.makedirs(output_folder, exist_ok=True)
        graph_save_path = os.path.join(output_folder, name_graph_saved + "_0.gexf")

    # Get open data.
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        exit(1)

    # Get the data from the json.
    df_kws_article = pd.DataFrame(columns=["year", "month", "day", "url", "title"] + segment_key)

    for year in Years:
        for month in range(1, 13):
            for day in range(1, 32):
                num = 0
                while True:
                    try:
                        data_tmp = data['data'][str(year)][str(month)][str(day)][num]
                        num += 1

                        base_data = {
                            "year": year,
                            "month": month,
                            "day": day,
                            "url": data_tmp.get("url", ""),
                            "title": data_tmp.get("title", "")
                        }

                        # Create a dictionary to hold the segment data
                        dict_tmp = {key: [] for key in segment_key}

                        # Extract segment data for each key
                        for key in segment_key:
                            if key in data_tmp:
                                for segment_data_tmp in data_tmp[key]:
                                    for segment_data in segment_data_tmp:
                                        dict_tmp[key].append(segment_data)
                            else:
                                dict_tmp[key].append(None)

                        max_length = max([len(value) for value in dict_tmp.values()])

                        for key in dict_tmp.keys():
                            while len(dict_tmp[key]) < max_length:
                                dict_tmp[key].append(None)

                        for i in range(max_length):
                            new_data = base_data.copy()
                            for key in segment_key:
                                new_data[key] = dict_tmp[key][i]
                            df_kws_article = pd.concat([df_kws_article, pd.DataFrame([new_data])], ignore_index=True)

                    except KeyError:
                        break
                    except Exception as e:
                        break

    # Create a dictionary to map words to their corresponding colors
    word_color_map = {}

    list_words = []

    # Iterate over rows to keep words from the same row grouped together
    for i, row in df_kws_article.iterrows():
        row_words = []

        for col_idx, val in enumerate(columns_to_use):
            words_in_column = row[val]

            if isinstance(words_in_column, list) and words_in_column:
                row_words += words_in_column

                for word in words_in_column:
                    word_color_map[word] = colors[col_idx % len(colors)]

        if row_words:
            list_words.append(row_words)

    # Generate the graph
    G = nx.Graph()

    for word_list in list_words:
        words = [word for word in word_list if word not in ignore_words]

        for word1, word2 in combinations(words, 2):
            if G.has_edge(word1, word2):
                G[word1][word2]['weight'] += 1
            else:
                G.add_edge(word1, word2, weight=1)

    # Set node colors as attributes (ensure colors are in hex format)
    nx.set_node_attributes(G, {word: {'color': color} for word, color in word_color_map.items()})

    # Save the graph with node colors in GEXF format
    if save_graph:
        nx.write_gexf(G, graph_save_path)