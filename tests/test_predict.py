"""Tests for the predict module."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from src.predict import predict
except ImportError:
    from predict import predict


class TestPredict:
    """Integration tests for the predict() function."""

    def test_predict_returns_dict(self):
        result = predict("Hello, how are you?")
        assert isinstance(result, dict)
        assert "label" in result
        assert "confidence" in result

    def test_label_values(self):
        result = predict("Free prize money click now")
        assert result["label"] in ("spam", "ham")

    def test_confidence_range(self):
        result = predict("Win a million dollars immediately")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_spam_detection(self):
        """A clearly spammy message should be classified as spam."""
        result = predict(
            "WINNER!! You have been selected to receive a $1000 cash prize! "
            "Call 09061701461 to claim NOW!"
        )
        assert result["label"] == "spam"

    def test_ham_detection(self):
        """A normal message should be classified as ham."""
        result = predict("Hey, are we still meeting for lunch tomorrow?")
        assert result["label"] == "ham"

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            predict("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            predict("   ")

    def test_very_long_input_raises(self):
        long_text = "a" * 20_000
        with pytest.raises(ValueError, match="maximum length"):
            predict(long_text)
