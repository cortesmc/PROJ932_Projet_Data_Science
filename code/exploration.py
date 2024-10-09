import json

# Charger le fichier JSON
with open('1eb80fb8b50.json', 'r') as f:
    data = json.load(f)

# Afficher un aperçu des données
print(data.keys())
print(json.dumps(data['metadata'], indent=2))  # Exemple d'exploration des métadonnées
