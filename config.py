import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
PIPELINES_DIR = os.path.join(PROJECT_ROOT, "pipelines")

RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "1eb80fb8b50.json")
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "processed_data.csv")
GRAPH_FILE = os.path.join(PROCESSED_DATA_DIR, "graph.gexf")
