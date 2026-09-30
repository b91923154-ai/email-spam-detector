"""
Training pipeline for the Email/SMS Spam Detection system.

Usage:
    python -m src.train
"""

import logging
import os
import pickle
import sys
import numpy as np
import pandas as pd

# Handle imports whether run directly or as package
try:
    from src.preprocessing import transform_text
    from src.model import NumPyTfidfVectorizer, NumPyMultinomialNB
except ImportError:
    from preprocessing import transform_text
    from model import NumPyTfidfVectorizer, NumPyMultinomialNB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "spam.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(PROJECT_ROOT, "spam.csv")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load dataset and return a cleaned DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at {path}")

    logger.info("Loading data from %s", path)
    df = pd.read_csv(path, encoding="latin-1")

    # Handle column variations in SMS Spam Collection dataset
    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]]
        df.columns = ["target", "text"]
    elif "target" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain either (v1, v2) or (target, text) columns.")

    # Encode target: 'spam' -> 1, 'ham' -> 0
    if df["target"].dtype == object:
        df["target"] = df["target"].astype(str).str.lower().map({"spam": 1, "ham": 0})
        df["target"] = df["target"].fillna(0).astype(int)

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    logger.info("Dropped %d duplicates (%d -> %d rows)", before - len(df), before, len(df))

    return df


def preprocess_column(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text preprocessing pipeline."""
    logger.info("Preprocessing text column...")
    df = df.copy()
    df["transformed_text"] = df["text"].apply(transform_text)
    return df


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame):
    """Train NumPyMultinomialNB model and return vectorizer and trained model."""
    logger.info("Building TF-IDF features...")
    vectorizer = NumPyTfidfVectorizer(max_features=4000, sublinear_tf=True)
    X = vectorizer.fit_transform(df["transformed_text"].values)
    y = df["target"].values

    # Train/Test Split (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    train_size = int(0.8 * len(df))
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    logger.info("Training Multinomial Naive Bayes model...")
    model = NumPyMultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    logger.info("Model Evaluation Metrics:")
    logger.info("  Accuracy  : %.4f", accuracy)
    logger.info("  Precision : %.4f", precision)
    logger.info("  Recall    : %.4f", recall)
    logger.info("  F1 Score  : %.4f", f1)

    print(f"\n--- Model Performance Summary ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    return model, vectorizer


def save_artifacts(model, vectorizer, model_path: str, vectorizer_path: str):
    """Persist model and vectoriser as pickle files."""
    # Save to models/
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model to %s", model_path)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Saved vectorizer to %s", vectorizer_path)

    # Also save copy to root for backwards compatibility if root exists
    root_model = os.path.join(PROJECT_ROOT, "model.pkl")
    root_vec = os.path.join(PROJECT_ROOT, "vectorizer.pkl")
    with open(root_model, "wb") as f:
        pickle.dump(model, f)
    with open(root_vec, "wb") as f:
        pickle.dump(vectorizer, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_and_clean_data(DATA_PATH)
    df = preprocess_column(df)

    best_model, vectorizer = train_and_evaluate(df)

    save_artifacts(best_model, vectorizer, MODEL_PATH, VECTORIZER_PATH)

    print(f"\n[SUCCESS] Training complete.")
    print(f"   Model saved to     : {MODEL_PATH}")
    print(f"   Vectorizer saved to: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
