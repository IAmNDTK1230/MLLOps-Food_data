import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pandas as pd

from src.ml.config import MLConfig, TrackingConfig
from src.ml.evaluate import evaluate_selected
from src.ml.registry import register_selected
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum


def test_selected_test_set_evaluated_once(monkeypatch, tmp_path):
    config = MLConfig(models_dir=tmp_path / "models", reports_dir=tmp_path / "reports")
    config.models_dir.mkdir()
    path = config.models_dir / "baseline.joblib"
    path.write_bytes(b"unit model double")
    atomic_json(
        config.models_dir / "selection.json",
        {
            "run_id": "unit-run",
            "dataset_hash": "unit-dataset",
            "split_file_sha256": "unit-split",
            "model_sha256": file_checksum(path),
            "tracking_uri": "sqlite:///state/mlflow.db",
        },
    )
    frame = pd.DataFrame(
        {"comment": ["unit heldout"], "sentiment": ["positive"], "split": ["test"]}
    )
    monkeypatch.setattr(
        "src.ml.evaluate.load_splits",
        lambda c: (frame, {"split_file_sha256": "unit-split"}, {"dataset_hash": "unit-dataset"}),
    )
    model = Mock()
    model.predict.return_value = ["positive"]
    monkeypatch.setattr("src.ml.evaluate.joblib.load", Mock(return_value=model))
    monkeypatch.setattr("src.ml.mlflow_tracking.configure_tracking", lambda c: MagicMock())
    first = evaluate_selected(config, TrackingConfig())
    second = evaluate_selected(config, TrackingConfig())
    assert first == second and first["training_run_id"] == "unit-run"
    model.predict.assert_called_once_with(["unit heldout"])
    # The train-stage output is immutable; DVC evaluate only writes its own outputs.
    assert "test_macro_f1" not in json.loads((config.models_dir / "selection.json").read_text())


def test_registry_records_provenance_but_never_promotes_alias(monkeypatch, tmp_path):
    config = MLConfig(models_dir=tmp_path / "models", reports_dir=tmp_path / "reports")
    atomic_json(
        config.models_dir / "selection.json",
        {
            "run_id": "unit-run",
            "model_uri": "runs:/unit-run/model",
            "dataset_hash": "unit-dataset",
            "validation_macro_f1": 0.5,
            "tracking_uri": "databricks://shopee",
        },
    )
    atomic_json(
        config.models_dir / "evaluation_state.json",
        {"training_run_id": "unit-run", "completed": True},
    )
    atomic_json(
        config.reports_dir / "test_metrics.json",
        {"training_run_id": "unit-run", "dataset_hash": "unit-dataset", "macro_f1": 0.4},
    )
    monkeypatch.setattr(
        "src.ml.registry.require_trainable", lambda c: (None, {"dataset_hash": "unit-dataset"})
    )
    tracking = Mock()
    tracking.register_model.return_value = SimpleNamespace(version="7")
    client = Mock()
    monkeypatch.setattr("src.ml.registry.mlflow", tracking)
    monkeypatch.setattr("src.ml.registry.MlflowClient", lambda: client)
    result = register_selected(
        config, TrackingConfig(tracking_uri="databricks://shopee"), "shopee_test"
    )
    tracking.register_model.assert_called_once_with(
        "runs:/unit-run/model", "shopee_test.ml.sentiment_classifier"
    )
    assert result["model_version"] == "7" and result["test_macro_f1"] == 0.4
    client.set_registered_model_alias.assert_not_called()
