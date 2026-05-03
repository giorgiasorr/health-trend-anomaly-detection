from pathlib import Path

import pandas as pd

# -----------------------------
# PATH SETUP
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "gdelt_with_topics.csv"
OUTPUT_DIR = BASE_DIR / "data" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_PATH = OUTPUT_DIR / "daily_counts.csv"
KEYWORD_PATH = OUTPUT_DIR / "keyword_trends.csv"
TOPIC_TIME_PATH = OUTPUT_DIR / "topic_over_time.csv"


# -----------------------------
# DATA LOADING
# -----------------------------


def load_data(path: Path) -> pd.DataFrame:
    """
    Load processed dataset.
    """
    return pd.read_csv(path)


# -----------------------------
# FEATURE 1: DAILY COUNTS
# -----------------------------


def build_daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes number of articles per day.
    """
    df["date_only"] = pd.to_datetime(df["date_only"])

    daily = df.groupby("date_only").size().reset_index(name="article_count")

    return daily


# -----------------------------
# FEATURE 2: KEYWORD TRENDS
# -----------------------------


def build_keyword_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes keyword frequency over time.
    """
    df["date_only"] = pd.to_datetime(df["date_only"])

    keyword_trends = df.groupby(["date_only", "keyword"]).size().unstack().fillna(0)

    return keyword_trends


# -----------------------------
# FEATURE 3: TOPIC OVER TIME
# -----------------------------


def build_topic_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes topic distribution over time.
    Requires dominant_topic column.
    """
    df["date_only"] = pd.to_datetime(df["date_only"])

    topic_time = df.groupby(["date_only", "dominant_topic"]).size().unstack().fillna(0)

    return topic_time


# -----------------------------
# SAVE FUNCTION
# -----------------------------


def save(df: pd.DataFrame, path: Path):
    """
    Save dataframe to CSV.
    """
    df.to_csv(path)
    print(f"Saved: {path}")


# -----------------------------
# MAIN PIPELINE
# -----------------------------


def main():

    print("Loading data...")
    df = load_data(INPUT_PATH)

    print(f"Initial rows: {len(df)}")

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------

    print("\nBuilding daily counts...")
    daily = build_daily_counts(df)
    save(daily, DAILY_PATH)

    print("\nBuilding keyword trends...")
    keywords = build_keyword_trends(df)
    save(keywords, KEYWORD_PATH)

    print("\nBuilding topic over time...")
    topic_time = build_topic_over_time(df)
    save(topic_time, TOPIC_TIME_PATH)

    print("\nFeature engineering completed.")


# -----------------------------
# RUN
# -----------------------------

if __name__ == "__main__":
    main()
