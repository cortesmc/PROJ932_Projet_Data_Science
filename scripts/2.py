from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np

import json
import pandas as pd
import string
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.express as px

from calendar import monthrange
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
# Load the data


def load_data(file_path, segment_key, years):
    """
    Load data from a JSON file and organize it into a DataFrame.

    Args:
        file_path (str): Path to the JSON file.
        segment_key (list): List of keys to extract from the JSON.
        years (list): List of years to extract data for.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return pd.DataFrame()

    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=["year", "month", "day", "url", "title"] + segment_key)

    # Extract data for each year, month, and day
    for year in years:
        for month in range(1, 13):
            # Get the number of days in the month
            _, num_days = monthrange(int(year), month)
            for day in range(1, num_days + 1):
                try:
                    # Check if the day exists in the data
                    if str(day) not in data['data'][str(year)][str(month)]:
                        continue
                    num = 0
                    while True:
                        try:
                            # Check if the article exists
                            if num >= len(data['data'][str(year)][str(month)][str(day)]):
                                break
                            data_tmp = data['data'][str(year)][str(month)][str(day)][num]
                            num += 1

                            # Create base data for the article
                            base_data = {
                                "year": year,
                                "month": month,
                                "day": day,
                                "url": data_tmp.get("url", ""),
                                "title": data_tmp.get("title", "")
                            }

                            # Extract segment data for each key
                            dict_tmp = {key: [] for key in segment_key}
                            for key in segment_key:
                                if key in data_tmp:
                                    for segment_data_tmp in data_tmp[key]:
                                        for segment_data in segment_data_tmp:
                                            dict_tmp[key].append(segment_data)
                                else:
                                    dict_tmp[key].append(None)

                            # Ensure all segments have the same length
                            max_length = max([len(value) for value in dict_tmp.values()])
                            for key in dict_tmp.keys():
                                while len(dict_tmp[key]) < max_length:
                                    dict_tmp[key].append(None)

                            # Add each segment to the DataFrame
                            for i in range(max_length):
                                new_data = base_data.copy()
                                for key in segment_key:
                                    new_data[key] = dict_tmp[key][i]
                                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

                        except KeyError:
                            break
                        except Exception as e:
                            print(f"Error processing data: {e}")
                            break
                except KeyError:
                    continue

    # Explode the 'content-segmented' column to handle lists
    df = df.explode("content-segmented", ignore_index=True)

    # Add a 'date' column for easier time-based analysis
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # Print some statistics
    number_articles = df.shape[0]
    none_count = df['content-segmented'].isna().sum()
    print(f'Number of articles: {number_articles}')
    print(f"Number of None type in content-segmented: {none_count}")
    print(f'Percentage of None value in total: {100 * none_count / df.shape[0]:.2f} %')

    # Drop rows with None in 'content-segmented'
    df = df.dropna(subset=['content-segmented'])

    return df

file_path = "../data/raw/1eb80fb8b50.json"
segment_key = ["kws-l", "loc-l", "org-l", "per-l", "content-segmented"]
years = ["2024"]

df = load_data(file_path, segment_key, years)
df.head()

# Étape 1 : Extraction des entités
# Concaténer les colonnes d'entités
entities = df["loc-l"].tolist() + df["org-l"].tolist() + df["per-l"].tolist()

# Aplatir la liste d'entités (résoudre le problème de sous-listes)
flattened_entities = [entity for sublist in entities for entity in (sublist if isinstance(sublist, list) else [sublist])]

# Supprimer les doublons
unique_entities = list(set(flattened_entities))

# Vérification du résultat
print(f"Nombre total d'entités uniques : {len(unique_entities)}")



# Étape 2 : Nettoyage des entités
entities = [e for e in entities if len(e) > 2]  # Supprimer les entités courtes ou non significatives

# Étape 3 : Embedding des entités
model = SentenceTransformer('all-MiniLM-L6-v2')  # Modèle pré-entraîné
entity_embeddings = model.encode(entities)

# Étape 4 : Calcul de la similarité
similarity_matrix = cosine_similarity(entity_embeddings)
threshold = 0.7  # Seuil de similarité
edges = np.argwhere(similarity_matrix > threshold)

# Étape 5 : Construction du graphe
G = nx.Graph()

# Ajouter les nœuds
for entity in entities:
    G.add_node(entity)

# Ajouter les arêtes basées sur la similarité
for edge in edges:
    i, j = edge
    if i != j:  # Éviter les boucles
        G.add_edge(entities[i], entities[j], weight=similarity_matrix[i, j])

# Étape 6 : Nettoyage du graphe
# Supprimer les nœuds isolés
isolated_nodes = list(nx.isolates(G))
G.remove_nodes_from(isolated_nodes)

# Résultats
print(f"Nombre de nœuds : {G.number_of_nodes()}")
print(f"Nombre d'arêtes : {G.number_of_edges()}")

# Export du graphe
nx.write_gexf(G, "semantic_similarity_graph.gexf")
