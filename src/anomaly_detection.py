from pathlib import Path

import pandas as pd

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "gdelt_with_topics.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "gdelt_with_anomalies.csv"
PLOT_DIR = BASE_DIR / "outputs" / "plots"


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path: Path) -> pd.DataFrame:
    """
    Load dataset with topics already assigned.
    Expected columns:
    - date_only
    - dominant_topic
    """
    return pd.read_csv(path)


# -----------------------------
# ANOMALY DETECTION (Z-SCORE)
# -----------------------------
def detect_anomalies(daily_counts: pd.Series, threshold: float = 2.0):
    """
    Detect anomalies using z-score method.
    """
    mean = daily_counts.mean()
    std = daily_counts.std()

    z_scores = (daily_counts - mean) / std
    anomalies = daily_counts[z_scores > threshold]

    return anomalies, z_scores


# -----------------------------
# SAVE OUTPUT
# -----------------------------
def save_output(df, anomalies, output_path):
    """
    Save dataset with anomaly flag for later analysis.
    """
    df["date_only"] = pd.to_datetime(df["date_only"])

    df["is_anomaly"] = df["date_only"].isin(anomalies.index)

    df.to_csv(output_path, index=False)


# -----------------------------
# PIPELINE
# -----------------------------
def run_anomaly_detection(input_path, output_path):
    """
    End-to-end anomaly detection pipeline:
    - Load data
    - Aggregate daily counts
    - Detect anomalies using z-score
    - Save results
    - Plot visualization
    """

    # Load full dataset (needed for saving + plotting context)
    df = load_data(input_path)

    # Ensure datetime format
    df["date_only"] = pd.to_datetime(df["date_only"])

    # Build daily time series
    daily_counts = df.groupby("date_only").size()

    # Detect anomalies
    anomalies, z_scores = detect_anomalies(daily_counts)

    print("\nDetected anomalies:")
    print(anomalies)

    # Save dataset with anomaly flag
    save_output(df, anomalies, output_path)

    print("\nSaved output to:", output_path)
    print("Saved plot to:", PLOT_DIR / "anomalies.png")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    run_anomaly_detection(INPUT_FILE, OUTPUT_FILE)
