"""
Root entry point for custom_model module. Delegates to src.model.
"""
from src.model import NumPyTfidfVectorizer, NumPyMultinomialNB  # noqa: F401

__all__ = ["NumPyTfidfVectorizer", "NumPyMultinomialNB"]
