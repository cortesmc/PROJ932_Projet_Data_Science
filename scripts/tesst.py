import pandas as pd

# Charger le fichier CSV
df = pd.read_csv("../data/processed/processed_data.csv")

# Afficher les 5 premières lignes pour voir les colonnes
print(df.head())

# Afficher les noms de colonnes
print(df.columns)
