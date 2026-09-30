"""
Root entry point for predict module. Delegates to src.predict.
"""
from src.predict import predict  # noqa: F401

__all__ = ["predict"]
