from config import RAW_DATA_FILE, PROCESSED_DATA_FILE
import json
import pandas as pd

def load_data(file_path):
    """
    Load JSON data from the specified file path.
    
    Args:
        file_path (str): Path to the JSON file.
    
    Returns:
        dict: Loaded JSON data or an empty dict if an error occurs.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return {}

def preprocess_data(data):
    """
    Preprocess the JSON data and convert it into a DataFrame.
    
    Args:
        data (dict): Raw JSON data.
    
    Returns:
        pd.DataFrame: Processed DataFrame containing article data.
    """
    df = pd.DataFrame(columns=["year", "month", "day", "url", "title", "loc-l", "org-l", "per-l", "content-segmented"])
    
    for year in data.get("data", {}):
        for month in data["data"][year]:
            for day in data["data"][year][month]:
                for article in data["data"][year][month][day]:
                    # Extract content-segmented properly
                    content_segments = []
                    if "content-segmented" in article:
                        for seg_group in article["content-segmented"]:
                            for seg in seg_group:
                                if isinstance(seg, list):
                                    content_segments.extend([str(item) for item in seg])  # Convert list elements to strings
                                else:
                                    content_segments.append(str(seg))  # Ensure each segment is a string

                    # Create entry for the DataFrame
                    entry = {
                        "year": year,
                        "month": month,
                        "day": day,
                        "url": article.get("url", ""),
                        "title": article.get("title", ""),
                        "loc-l": article.get("loc-l", []),
                        "org-l": article.get("org-l", []),
                        "per-l": article.get("per-l", []),
                        "content-segmented": " ".join(content_segments),  # Join segments into a single string
                    }
                    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

    return df

if __name__ == "__main__":
    # Load and preprocess the data
    data = load_data("../data/raw/1eb80fb8b50.json")
    df = preprocess_data(data)

    # Save the processed data to a CSV file
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"✅ Données enregistrées : {PROCESSED_DATA_FILE}")
