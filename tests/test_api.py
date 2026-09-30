"""Tests for the FastAPI backend (api.py)."""

import sys
import os

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from src.api import app
except ImportError:
    from api import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert "supported_labels" in data

    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestPredictEndpoint:
    def test_valid_prediction(self):
        response = client.post(
            "/predict",
            json={"text": "WINNER!! You have been selected to receive a $1000 cash prize!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "confidence" in data
        assert data["label"] in ("spam", "ham")
        assert 0.0 <= data["confidence"] <= 1.0

    def test_ham_message(self):
        response = client.post(
            "/predict",
            json={"text": "Hi, can we reschedule our meeting to 3 PM?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "ham"

    def test_empty_text_returns_422(self):
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_missing_text_field_returns_422(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_invalid_method_returns_405(self):
        response = client.get("/predict")
        assert response.status_code == 405


class TestBatchPredictEndpoint:
    def test_batch_prediction(self):
        response = client.post(
            "/predict/batch",
            json={
                "texts": [
                    "WINNER!! You have won $1000 cash prize! Call 09061701461 to claim NOW!",
                    "Hey, are we still meeting for lunch tomorrow?"
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
        assert data["results"][0]["label"] == "spam"
        assert data["results"][1]["label"] == "ham"
