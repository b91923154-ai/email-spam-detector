"""
Email & SMS Spam Detection Package.
"""

from .preprocessing import transform_text
from .model import NumPyTfidfVectorizer, NumPyMultinomialNB
from .predict import predict

__all__ = [
    "transform_text",
    "NumPyTfidfVectorizer",
    "NumPyMultinomialNB",
    "predict",
]
