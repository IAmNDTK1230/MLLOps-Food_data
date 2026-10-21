import json
from unittest.mock import Mock

import numpy as np
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.model_loader import LoadedModel, load_model
from src.api.monitoring import InferenceMonitor


def test_prediction_probabilities_version_and_private_text_not_logged(tmp_path):
    pipeline = Mock()
    pipeline.classes_ = np.array(["negative", "neutral", "positive"])
    pipeline.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
    monitor = InferenceMonitor(tmp_path / "inference.jsonl")
    with TestClient(create_app(lambda: LoadedModel(pipeline, "unit-version"), monitor)) as client:
        response = client.post("/predict", json={"text": "Private unit input"})
        assert response.status_code == 200
        assert response.json() == {
            "sentiment": "positive",
            "probabilities": {"negative": 0.1, "neutral": 0.2, "positive": 0.7},
            "model_version": "unit-version",
        }
        assert client.get("/health").json()["model_version"] == "unit-version"
        assert client.get("/monitoring").json()["requests"] == 1
        assert client.post("/predict", json={"text": "  "}).status_code == 422
    contents = monitor.path.read_text()
    assert "Private unit input" not in contents
    assert set(json.loads(contents)) == {
        "timestamp",
        "model_version",
        "predicted_class",
        "confidence",
        "text_length",
    }


def test_no_model_returns_unavailable_without_secret_details(tmp_path):
    with TestClient(
        create_app(
            Mock(side_effect=RuntimeError("token=secret")), InferenceMonitor(tmp_path / "log")
        )
    ) as client:
        assert client.get("/health").status_code == 503
        response = client.post("/predict", json={"text": "unit"})
        assert response.status_code == 503 and "secret" not in response.text


def test_mlflow_configuration_never_falls_back_to_local(monkeypatch):
    monkeypatch.setattr("src.api.model_loader.load_dotenv", lambda: None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks://shopee")
    monkeypatch.delenv("MLFLOW_MODEL_URI", raising=False)
    monkeypatch.setenv("LOCAL_MODEL_PATH", "do-not-load.joblib")
    import pytest

    with pytest.raises(ValueError, match="local fallback is disabled"):
        load_model()
