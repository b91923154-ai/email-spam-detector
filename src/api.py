"""
FastAPI backend for the Email/SMS Spam Detection system.

Endpoints
---------
GET  /         – root info endpoint
GET  /health   – health-check
POST /predict  – classify a single text as spam or ham
POST /predict/batch – classify a batch of text messages

Usage:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import logging
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from src.predict import predict
except ImportError:
    from predict import predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Email & SMS Spam Detection API",
    description="High-performance, production-ready REST API for classifying email and SMS messages as spam or ham.",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Data Transfer Objects (DTOs)
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Email / SMS body text")


class PredictResponse(BaseModel):
    label: str
    confidence: float


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of messages to classify")


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


class HealthResponse(BaseModel):
    status: str


class ServiceInfoResponse(BaseModel):
    name: str
    version: str
    status: str
    supported_labels: List[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=ServiceInfoResponse)
async def root():
    """Root metadata endpoint."""
    return ServiceInfoResponse(
        name="Email & SMS Spam Detection API",
        version="1.0.0",
        status="active",
        supported_labels=["ham", "spam"],
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health-check endpoint."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """Classify input text as spam or ham."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    try:
        result = predict(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal server error")

    return PredictResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch_endpoint(request: BatchPredictRequest):
    """Classify a batch of messages."""
    results = []
    for text in request.texts:
        cleaned_text = text.strip()
        if not cleaned_text:
            results.append(PredictResponse(label="ham", confidence=1.0))
            continue
        try:
            res = predict(cleaned_text)
            results.append(PredictResponse(**res))
        except Exception as exc:
            logger.warning("Error predicting text batch item: %s", exc)
            results.append(PredictResponse(label="unknown", confidence=0.0))

    return BatchPredictResponse(results=results)
