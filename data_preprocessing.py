"""
Root entry point for data_preprocessing module. Delegates to src.preprocessing.
"""
from src.preprocessing import transform_text  # noqa: F401

__all__ = ["transform_text"]
