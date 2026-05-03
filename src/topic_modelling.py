from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "gdelt_processed.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "gdelt_with_topics.csv"


# -----------------------------
# DATA LOADING
# -----------------------------
def load_data(path: Path) -> pd.DataFrame:
    """
    Load preprocessed dataset.
    """
    return pd.read_csv(path)


# -----------------------------
# TEXT PREPARATION
# -----------------------------
def prepare_text(df: pd.DataFrame, text_col="title"):
    """
    Clean and prepare text for vectorization.
    """
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    texts = df[text_col].astype(str).str.lower()
    return df, texts


# -----------------------------
# VECTORISATION (TF-IDF)
# -----------------------------
def vectorize(texts, max_features=1000):
    """
    Convert text into TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")

    X = vectorizer.fit_transform(texts)
    return X, vectorizer


# -----------------------------
# TOPIC MODEL (LDA)
# -----------------------------
def train_lda(X, n_topics=5, random_state=42):
    """
    Train LDA topic model.
    """
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)

    lda.fit(X)
    return lda


# -----------------------------
# TOPIC ASSIGNMENT
# -----------------------------
def assign_topics(df, lda, X):
    """
    Assign dominant topic per document.
    """
    topic_distribution = lda.transform(X)
    df["dominant_topic"] = topic_distribution.argmax(axis=1)
    return df, topic_distribution


# -----------------------------
# TOPIC INSPECTION
# -----------------------------
def print_topics(lda, vectorizer, n_words=8):
    """
    Print interpretable topics.
    """
    words = vectorizer.get_feature_names_out()

    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[: -n_words - 1 : -1]]
        print(f"\nTopic {idx}: {', '.join(top_words)}")


# -----------------------------
# FULL PIPELINE
# -----------------------------
def run_topic_modeling(input_path, output_path):
    """
    End-to-end LDA topic modeling pipeline:
    - TF-IDF vectorization
    - Latent Dirichlet Allocation
    - Topic assignment per document
    - Saving enriched dataset
    """

    # 1. Load data
    df = load_data(input_path)

    # 2. Prepare text
    df, texts = prepare_text(df)

    # 3. Vectorize
    X, vectorizer = vectorize(texts)

    # 4. Train model
    lda = train_lda(X)

    # 5. Assign topics
    df, topic_distribution = assign_topics(df, lda, X)

    # 6. Save dataset
    df.to_csv(output_path, index=False)

    # 7. Save models (reproducibility)
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    joblib.dump(lda, models_dir / "lda_model.pkl")
    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.pkl")

    # 8. Output
    print("Saved topic-enhanced dataset to:", output_path)
    print("\nTop topics:")
    print_topics(lda, vectorizer)

    return df, lda, vectorizer, topic_distribution


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    run_topic_modeling(INPUT_FILE, OUTPUT_FILE)
