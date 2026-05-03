from pathlib import Path

import pandas as pd

# -----------------------------
# PATH SETUP
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "gdelt_raw.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "gdelt_processed.csv"

# -----------------------------
# LOAD DATA
# -----------------------------


def load_data(path):
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    return df


# -----------------------------
# CLEAN DATA
# -----------------------------


def clean_data(df):
    print("\nCleaning data...")

    # Remove rows with missing titles or dates
    df = df.dropna(subset=["title", "date"])

    # Remove duplicates based on title + date
    df = df.drop_duplicates(subset=["title", "date"])

    # Normalize text
    df["title"] = df["title"].str.lower()

    return df


# -----------------------------
# DATE PROCESSING
# -----------------------------


def process_dates(df):
    print("\nProcessing dates...")

    # Convert GDELT date format to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows where date conversion failed
    df = df.dropna(subset=["date"])

    # Create additional time features
    df["date_only"] = df["date"].dt.date
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week

    return df


# -----------------------------
# FILTER HEALTH-RELATED CONTENT (optional but useful)
# -----------------------------


def filter_health_articles(df):
    print("\nFiltering health-related articles...")

    health_keywords = [
        "flu",
        "virus",
        "infection",
        "disease",
        "outbreak",
        "epidemic",
        "pandemic",
        "antibiotic",
        "resistance",
        "hospital",
    ]

    pattern = "|".join(health_keywords)

    df = df[df["title"].str.contains(pattern, na=False)]

    print(f"Remaining articles after filtering: {len(df)}")

    return df


# -----------------------------
# SAVE DATA
# -----------------------------


def save_data(df, path):
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"\nSaved processed data to: {path}")


# -----------------------------
# MAIN PIPELINE
# -----------------------------


def main():
    df = load_data(RAW_DATA_PATH)

    print(f"\nInitial dataset size: {len(df)}")

    df = clean_data(df)
    print(f"After cleaning: {len(df)}")

    df = process_dates(df)
    print(f"After date processing: {len(df)}")

    # df = filter_health_articles(df) # too strict of a filter

    print("\nPreview:")
    print(df.head())

    save_data(df, OUTPUT_PATH)


# -----------------------------
# RUN SCRIPT
# -----------------------------

if __name__ == "__main__":
    main()
