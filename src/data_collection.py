import time
from pathlib import Path

import pandas as pd
import requests

# -----------------------------
# PATH SETUP
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_DIR / "gdelt_raw.csv"

# -----------------------------
# CONFIGURATION
# -----------------------------

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

KEYWORDS = [
    "flu",
    "virus",
    "outbreak",
    "infection",
    "epidemic",
    "pandemic",
    "covid",
    "antimicrobial resistance",
    "antibiotic resistance",
    "hospital",
    "disease",
]

START_DATE = "20240101000000"  # format: YYYYMMDDHHMMSS
END_DATE = "20260101000000"

MAX_RECORDS = 250  # per keyword
SLEEP_TIME = 5  # seconds between requests

# -----------------------------
# FUNCTION: FETCH DATA
# -----------------------------


def fetch_articles(keyword, start_date, end_date, max_records=250):
    """
    Fetch articles from GDELT API for a given keyword.
    """
    params = {
        "query": keyword,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": start_date,
        "enddatetime": end_date,
        "maxrecords": max_records,
        "sort": "datedesc",
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "articles" not in data:
            return []

        return data["articles"]

    except Exception as e:
        print(f"Error fetching '{keyword}': {e}")

        if "429" in str(e):
            print("Rate limit hit → sleeping 5 seconds")
            time.sleep(7)

        return []


# -----------------------------
# MAIN COLLECTION FUNCTION
# -----------------------------


def collect_data():
    """
    Collect data for all keywords and return a DataFrame.
    """
    all_records = []

    for keyword in KEYWORDS:
        print(f"\nFetching data for: {keyword}")

        articles = fetch_articles(keyword, START_DATE, END_DATE, MAX_RECORDS)

        if not articles:
            print(f"  -> No data for {keyword}")
            continue

        print(f"  -> Retrieved {len(articles)} articles")

        for article in articles:
            record = {
                "keyword": keyword,
                "title": article.get("title"),
                "date": article.get("seendate"),
                "url": article.get("url"),
                "source_country": article.get("sourceCountry"),
                "domain": article.get("domain"),
            }
            all_records.append(record)

        time.sleep(SLEEP_TIME)

    df = pd.DataFrame(all_records)
    return df


# -----------------------------
# SAVE FUNCTION
# -----------------------------


def save_data(df, path):
    """
    Save DataFrame to CSV.
    """
    df.to_csv(str(path), index=False, encoding="utf-8")
    print(f"\nSaved data to: {path}")


# -----------------------------
# MAIN EXECUTION
# -----------------------------

if __name__ == "__main__":
    print("Starting GDELT data collection...\n")

    df = collect_data()

    print(f"\nTotal records collected: {len(df)}")

    # Basic check
    print("\nPreview:")
    print(df.head())

    save_data(df, OUTPUT_PATH)

    print("\nData collection complete.")
