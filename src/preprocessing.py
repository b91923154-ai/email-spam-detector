"""
Shared text preprocessing for the Email/SMS Spam Detection system.

This module is used by both training and prediction to ensure
consistent text transformation.
"""

import string
import logging
import re
import nltk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK bootstrap – download required data once safely
# ---------------------------------------------------------------------------

def ensure_nltk_data() -> None:
    """Download NLTK resources safely if missing."""
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    for resource in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.download(resource, quiet=True)
        except Exception as exc:
            logger.warning("Note downloading NLTK resource %s: %s", resource, exc)


ensure_nltk_data()

_stemmer = nltk.stem.PorterStemmer()

try:
    from nltk.corpus import stopwords
    _stop_words = set(stopwords.words("english"))
except Exception:
    _stop_words = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
        "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
        "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
        "about", "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    }

_punctuation = set(string.punctuation)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform_text(text: str) -> str:
    """Clean and normalise a single text string.

    Steps
    -----
    1. Lower-case
    2. Tokenise (NLTK word_tokenize with regex fallback)
    3. Keep only alphanumeric tokens
    4. Remove English stop-words and punctuation characters
    5. Apply Porter stemming

    Parameters
    ----------
    text : str
        Raw email / SMS body text.

    Returns
    -------
    str
        Space-joined cleaned tokens.
    """
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    text = text.lower()

    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = re.findall(r"\b\w+\b", text)

    # Keep only alphanumeric tokens
    tokens = [t for t in tokens if t.isalnum()]

    # Remove stop-words and punctuation
    tokens = [t for t in tokens if t not in _stop_words and t not in _punctuation]

    # Stem
    tokens = [_stemmer.stem(t) for t in tokens]

    return " ".join(tokens)
