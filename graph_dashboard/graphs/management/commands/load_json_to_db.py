import json
from datetime import datetime
from django.core.management.base import BaseCommand
from graphs.models import Node

class Command(BaseCommand):
    help = "Charge les données d'un fichier JSON dans la base de données."

    def add_arguments(self, parser):
        parser.add_argument('json_file', type=str, help="Chemin vers le fichier JSON.")

    def handle(self, *args, **kwargs):
        json_file_path = kwargs['json_file']

        # Charger le fichier JSON
        with open(json_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Parcourir les données et insérer dans la base de données
        for year, months in data["data"].items():
            for month, days in months.items():
                for day, records in days.items():
                    for record in records:
                        timestamp = record.get("timestamp", None)
                        if timestamp:
                            timestamp = datetime.fromtimestamp(timestamp)

                        # Ajouter les nœuds de type "person"
                        if "per" in record:
                            for person in record["per"]:
                                node, created = Node.objects.get_or_create(
                                    name=person,
                                    defaults={
                                        'node_type': 'person',
                                        'timestamp': timestamp,
                                    }
                                )
                                if not created:
                                    node.timestamp = timestamp
                                    node.save()

                        # Ajouter les nœuds de type "location"
                        if "loc" in record:
                            for location in record["loc"]:
                                node, created = Node.objects.get_or_create(
                                    name=location,
                                    defaults={
                                        'node_type': 'location',
                                        'timestamp': timestamp,
                                    }
                                )
                                if not created:
                                    node.timestamp = timestamp
                                    node.save()

                        # Ajouter les nœuds de type "organization"
                        if "org" in record:
                            for organization in record["org"]:
                                node, created = Node.objects.get_or_create(
                                    name=organization,
                                    defaults={
                                        'node_type': 'organization',
                                        'timestamp': timestamp,
                                    }
                                )
                                if not created:
                                    node.timestamp = timestamp
                                    node.save()

        self.stdout.write(self.style.SUCCESS("Données insérées avec succès !"))
