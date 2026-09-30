"""
Prediction module for the Email/SMS Spam Detection system.

The model and vectoriser are loaded **once** when this module is first
imported or initialized. Subsequent calls to ``predict()`` reuse them.
"""

import logging
import os
import pickle
import numpy as np

# Flexible imports
try:
    from src.preprocessing import transform_text
    from src.model import NumPyTfidfVectorizer, NumPyMultinomialNB
except ImportError:
    try:
        from preprocessing import transform_text
        from model import NumPyTfidfVectorizer, NumPyMultinomialNB
    except ImportError:
        from custom_model import NumPyTfidfVectorizer, NumPyMultinomialNB
        from data_preprocessing import transform_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.pkl")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, "model.pkl")

VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "vectorizer.pkl")
if not os.path.exists(VECTORIZER_PATH):
    VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "vectorizer.pkl")

# ---------------------------------------------------------------------------
# Load model and vectoriser once
# ---------------------------------------------------------------------------

_model = None
_vectorizer = None


def _load_pickle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}. "
            "Please run 'python train.py' first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _init_artifacts():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        _model = _load_pickle(MODEL_PATH)
        _vectorizer = _load_pickle(VECTORIZER_PATH)
        logger.info("Model and vectorizer loaded successfully.")


# Initialize artifacts on module import if files exist
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        _init_artifacts()
except Exception as exc:
    logger.warning("Could not auto-load artifacts on import: %s", exc)


# ---------------------------------------------------------------------------
# Label mapping & constants
# ---------------------------------------------------------------------------

_LABEL_MAP = {0: "ham", 1: "spam"}
MAX_INPUT_LENGTH = 10_000  # characters


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(text: str) -> dict:
    """Classify a single text as *spam* or *ham*.

    Parameters
    ----------
    text : str
        Raw email / SMS body text.

    Returns
    -------
    dict
        ``{"label": "spam" | "ham", "confidence": float}``

    Raises
    ------
    ValueError
        If *text* is empty or exceeds ``MAX_INPUT_LENGTH``.
    """
    if not text or not str(text).strip():
        raise ValueError("Input text must not be empty.")

    text_str = str(text)

    if len(text_str) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"Input text exceeds maximum length of {MAX_INPUT_LENGTH} characters."
        )

    _init_artifacts()

    # 1. Preprocess (same pipeline as training)
    cleaned = transform_text(text_str)

    # 2. Vectorise
    vector = _vectorizer.transform([cleaned])

    # 3. Predict
    prediction = int(_model.predict(vector)[0])
    label = _LABEL_MAP.get(prediction, "ham")

    # 4. Confidence calculation
    if hasattr(_model, "predict_proba"):
        probabilities = _model.predict_proba(vector)[0]
        confidence = float(np.max(probabilities))
    else:
        confidence = 1.0

    return {"label": label, "confidence": confidence}
