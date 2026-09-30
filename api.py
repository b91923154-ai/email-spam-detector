"""
Root entry point for FastAPI API server. Delegates to src.api.
"""
from src.api import app, PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse  # noqa: F401

__all__ = ["app", "PredictRequest", "PredictResponse", "BatchPredictRequest", "BatchPredictResponse"]
