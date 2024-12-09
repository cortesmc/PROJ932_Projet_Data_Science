import json
import pandas as pd
from datetime import datetime
from django.core.management.base import BaseCommand
from graphs.models import Node

class Command(BaseCommand):
    help = "Extract data from a JSON file and store it in the database."

    def add_arguments(self, parser):
        parser.add_argument('json_file', type=str, help="Path to the JSON file.")

    def handle(self, *args, **kwargs):
        json_file_path = kwargs['json_file']

        # Load JSON file
        try:
            with open(json_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f"File not found: {json_file_path}"))
            return
        except json.JSONDecodeError as e:
            self.stdout.write(self.style.ERROR(f"JSON decoding error: {e}"))
            return

        segment_key = ["kws-l", "loc-l", "org-l", "per-l"]
        years = data["data"].keys()

        df_kws_article = pd.DataFrame(columns=["year", "month", "day", "url", "title"] + segment_key)

        # Extract data
        for year in years:
            for month in range(1, 13):
                for day in range(1, 32):
                    num = 0
                    while True:
                        try:
                            record = data['data'][str(year)][str(month)][str(day)][num]
                            num += 1

                            base_data = {
                                "year": year,
                                "month": month,
                                "day": day,
                                "url": record.get("url", ""),
                                "title": record.get("title", ""),
                            }

                            # Create a dictionary to hold the segment data
                            dict_tmp = {key: [] for key in segment_key}

                            for key in segment_key:
                                if key in record:
                                    dict_tmp[key].extend(record[key])
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
                            self.stdout.write(self.style.WARNING(f"Error processing record: {e}"))
                            break

        # Insert into database
        for _, row in df_kws_article.iterrows():
            timestamp = datetime(
                year=int(row["year"]),
                month=int(row["month"]),
                day=int(row["day"])
            )

            # Add nodes for "person"
            if pd.notna(row["per-l"]):
                node, created = Node.objects.get_or_create(
                    name=row["per-l"],
                    defaults={
                        "node_type": "person",
                        "timestamp": timestamp,
                    }
                )
                if not created:
                    node.timestamp = timestamp
                    node.save()

            # Add nodes for "location"
            if pd.notna(row["loc-l"]):
                node, created = Node.objects.get_or_create(
                    name=row["loc-l"],
                    defaults={
                        "node_type": "location",
                        "timestamp": timestamp,
                    }
                )
                if not created:
                    node.timestamp = timestamp
                    node.save()

            # Add nodes for "organization"
            if pd.notna(row["org-l"]):
                node, created = Node.objects.get_or_create(
                    name=row["org-l"],
                    defaults={
                        "node_type": "organization",
                        "timestamp": timestamp,
                    }
                )
                if not created:
                    node.timestamp = timestamp
                    node.save()

        self.stdout.write(self.style.SUCCESS("Data successfully inserted into the database!"))
